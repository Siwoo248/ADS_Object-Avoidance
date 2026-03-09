"""
YOLO Obstacle Avoidance Web Viewer

Serves a web interface on port 8082 (HTTP) and 8083 (WebSocket)
for viewing YOLO detection frames and adjusting obstacle avoidance parameters.

Runs inside the same process as the YOLO detector.
"""

import asyncio
import json
import time
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, Lock
from typing import Optional, Set, Dict, Callable

import websockets


# ── Whitelisted parameters ──────────────────────────────────────

AVOIDANCE_PARAMS = {
    'LEFT_LANE_X': int,
    'RIGHT_LANE_X': int,
    'CRITICAL_DISTANCE': float,
    'DECISION_ENTER': float,
    'DECISION_EXIT': float,
    'MICRO_ADJUST_THRESHOLD': float,
    'MICRO_ADJUST_STEERING': float,
    'LANE_CHANGE_STEERING': float,
    'THROTTLE': float,
    'PREFERRED_LANE': str,
    'LANE_CHANGE_PASS_DISTANCE': float,
    'VEHICLE_SPEED': float,
    'LATERAL_SPEED': float,
}

YOLO_PARAMS = {
    'conf_threshold': float,
    'iou_threshold': float,
}


# ── WebSocket Server ────────────────────────────────────────────

class YOLOWebSocketServer:
    """Async WebSocket server for frame streaming and parameter control."""

    def __init__(self, port: int, on_parameter: Callable, on_action: Callable,
                 on_manual_control: Callable):
        self.port = port
        self._on_parameter = on_parameter
        self._on_action = on_action
        self._on_manual_control = on_manual_control

        self.clients: Set = set()
        self._lock = Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._server = None
        self._ready = False
        self._send_futures: Dict[int, asyncio.Future] = {}

    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        waited = 0.0
        while not self._ready and waited < 5:
            time.sleep(0.1)
            waited += 0.1

    def stop(self):
        if not self._loop or not self._loop.is_running():
            return

        async def _shutdown():
            with self._lock:
                clients = list(self.clients)
                self.clients.clear()
            close_tasks = [asyncio.ensure_future(c.close()) for c in clients]
            if close_tasks:
                await asyncio.wait(close_tasks, timeout=2)
            if self._server:
                self._server.close()
                await self._server.wait_closed()
            self._loop.stop()

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            future.result(timeout=5)
        except Exception:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def client_count(self) -> int:
        return len(self.clients)

    def broadcast_binary(self, data: bytes):
        if not self._loop:
            return
        with self._lock:
            dead = set()
            for client in self.clients:
                try:
                    cid = id(client)
                    prev = self._send_futures.get(cid)
                    if prev is not None and not prev.done():
                        continue
                    future = asyncio.run_coroutine_threadsafe(
                        client.send(data), self._loop,
                    )
                    self._send_futures[cid] = future
                except Exception:
                    dead.add(client)
            for client in dead:
                self._send_futures.pop(id(client), None)
            self.clients -= dead

    def broadcast_text(self, message: str):
        if not self._loop:
            return
        with self._lock:
            dead = set()
            for client in self.clients:
                try:
                    asyncio.run_coroutine_threadsafe(client.send(message), self._loop)
                except Exception:
                    dead.add(client)
            self.clients -= dead

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _start():
            try:
                server = await websockets.serve(
                    self._ws_handler, '0.0.0.0', self.port,
                    ping_interval=20, ping_timeout=20,
                )
                print(f"[YOLO-WS] Server started on port {self.port}")
                self._ready = True
                return server
            except Exception as e:
                print(f"[YOLO-WS] Startup error: {e}")
                self._ready = False
                return None

        try:
            self._server = self._loop.run_until_complete(_start())
            if self._server:
                self._loop.run_forever()
        except Exception as e:
            print(f"[YOLO-WS] Server error: {e}")
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _ws_handler(self, websocket):
        with self._lock:
            self.clients.add(websocket)

        addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"[YOLO-WS] Client connected: {addr}")

        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                    msg_type = data.get('type')

                    if msg_type == 'parameter':
                        self._on_parameter(data.get('name'), data.get('value'))

                    elif msg_type == 'action':
                        self._on_action(data.get('action'))

                    elif msg_type == 'manual_control':
                        self._on_manual_control(
                            data.get('steering', 0.0),
                            data.get('throttle', 0.0),
                        )

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[YOLO-WS] Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            cid = id(websocket)
            with self._lock:
                self.clients.discard(websocket)
            self._send_futures.pop(cid, None)
            print(f"[YOLO-WS] Client disconnected: {addr}")


# ── HTTP Server ─────────────────────────────────────────────────

class YOLOHTTPServer:
    """Threaded HTTP server serving the YOLO viewer page."""

    def __init__(self, port: int, ws_port: int, on_parameter: Callable,
                 on_action: Callable, get_config: Callable):
        self.port = port
        self.ws_port = ws_port
        self._on_parameter = on_parameter
        self._on_action = on_action
        self._get_config = get_config
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[Thread] = None

    def start(self):
        on_param = self._on_parameter
        on_action = self._on_action
        get_config = self._get_config
        ws_port = self.ws_port

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                msg = fmt % args
                if "code 404" in msg or "code 500" in msg:
                    print(f"[YOLO-HTTP] {msg}")

            def do_GET(self):
                if self.path == '/':
                    self._serve_html()
                elif self.path == '/config':
                    self._serve_config()
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK\n')
                elif self.path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == '/parameter':
                    self._handle_parameter()
                elif self.path == '/action':
                    self._handle_action()
                else:
                    self.send_error(404)

            def _serve_html(self):
                html_path = Path(__file__).parent / 'yolo_viewer.html'
                try:
                    with open(html_path, 'r') as f:
                        template = f.read()
                    cfg = get_config()
                    html = template.replace('{{WS_PORT}}', str(ws_port))
                    for key, val in cfg.items():
                        html = html.replace('{{' + key + '}}', str(val))
                    payload = html.encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(f"Error: {e}".encode())

            def _serve_config(self):
                cfg = get_config()
                payload = json.dumps(cfg).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

            def _handle_parameter(self):
                try:
                    length = int(self.headers['Content-Length'])
                    data = json.loads(self.rfile.read(length).decode())
                    on_param(data.get('name'), data.get('value'))
                    self._json_response(200, {'status': 'ok', 'name': data.get('name'),
                                               'value': data.get('value')})
                except Exception as e:
                    self._json_response(500, {'status': 'error', 'message': str(e)})

            def _handle_action(self):
                try:
                    length = int(self.headers['Content-Length'])
                    data = json.loads(self.rfile.read(length).decode())
                    on_action(data.get('action'))
                    self._json_response(200, {'status': 'ok', 'action': data.get('action')})
                except Exception as e:
                    self._json_response(500, {'status': 'error', 'message': str(e)})

            def _json_response(self, code, data):
                payload = json.dumps(data).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

        def _serve():
            try:
                self._server.serve_forever()
            except Exception as e:
                print(f"[YOLO-HTTP] Server error: {e}")

        try:
            self._server = ThreadingHTTPServer(('0.0.0.0', self.port), Handler)
            self._thread = Thread(target=_serve, daemon=True)
            self._thread.start()
            time.sleep(0.2)
            print(f"[YOLO-HTTP] Server started on port {self.port}")
        except Exception as e:
            print(f"[YOLO-HTTP] Failed to start: {e}")

    def stop(self):
        if self._server:
            self._server.shutdown()


# ── Orchestrator ────────────────────────────────────────────────

class YOLOWebViewer:
    """
    Orchestrates HTTP + WebSocket servers for the YOLO viewer.

    Usage:
        viewer = YOLOWebViewer(http_port=8082, avoidance=avoidance_system, detector=detector)
        viewer.start()
        # In loop:
        viewer.broadcast_frame(annotated_frame)
        viewer.broadcast_status({...})
        # On shutdown:
        viewer.stop()
    """

    def __init__(self, http_port: int = 8082, avoidance=None, detector=None):
        self.http_port = http_port
        self.ws_port = http_port + 1
        self.avoidance = avoidance
        self.detector = detector

        self._last_frame_time = 0
        self._frame_interval = 1.0 / 30  # max 30 FPS to browser
        self._last_status_time = 0
        self._status_interval = 1.0  # 1 Hz status updates

        # Manual control state
        self.manual_mode = False
        self.manual_steering = 0.0
        self.manual_throttle = 0.0

        self.ws_server = YOLOWebSocketServer(
            port=self.ws_port,
            on_parameter=self._handle_parameter,
            on_action=self._handle_action,
            on_manual_control=self._handle_manual_control,
        )
        self.http_server = YOLOHTTPServer(
            port=self.http_port,
            ws_port=self.ws_port,
            on_parameter=self._handle_parameter,
            on_action=self._handle_action,
            get_config=self._get_config,
        )

    def start(self):
        self.http_server.start()
        self.ws_server.start()
        print(f"\n[YOLO Viewer] Web interface: http://localhost:{self.http_port}")
        print(f"[YOLO Viewer] WebSocket: ws://localhost:{self.ws_port}\n")

    def stop(self):
        self.ws_server.stop()
        self.http_server.stop()
        print("[YOLO Viewer] Stopped")

    def broadcast_frame(self, frame):
        """JPEG-encode and broadcast a frame to all WebSocket clients."""
        now = time.time()
        if now - self._last_frame_time < self._frame_interval:
            return
        self._last_frame_time = now

        if self.ws_server.client_count == 0:
            return

        success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            self.ws_server.broadcast_binary(buf.tobytes())

    def broadcast_status(self, status: dict):
        """Broadcast status JSON to all WebSocket clients (rate-limited to 1 Hz)."""
        now = time.time()
        if now - self._last_status_time < self._status_interval:
            return
        self._last_status_time = now

        if self.ws_server.client_count == 0:
            return

        status['type'] = 'status'
        self.ws_server.broadcast_text(json.dumps(status))

    def _handle_parameter(self, name: str, value):
        """Apply a parameter update to the avoidance system or detector."""
        if name in AVOIDANCE_PARAMS and self.avoidance is not None:
            cast = AVOIDANCE_PARAMS[name]
            try:
                typed_value = cast(value)
                setattr(self.avoidance, name, typed_value)
                # Update derived values
                if name in ('LEFT_LANE_X', 'RIGHT_LANE_X'):
                    self.avoidance.LANE_WIDTH_PIXELS = (
                        self.avoidance.RIGHT_LANE_X - self.avoidance.LEFT_LANE_X
                    )
                print(f"[YOLO Viewer] {name} = {typed_value}")
            except (ValueError, TypeError) as e:
                print(f"[YOLO Viewer] Invalid value for {name}: {e}")

        elif name in YOLO_PARAMS and self.detector is not None:
            cast = YOLO_PARAMS[name]
            try:
                typed_value = cast(value)
                setattr(self.detector, name, typed_value)
                print(f"[YOLO Viewer] {name} = {typed_value}")
            except (ValueError, TypeError) as e:
                print(f"[YOLO Viewer] Invalid value for {name}: {e}")

        else:
            print(f"[YOLO Viewer] Unknown parameter: {name}")

    def _handle_action(self, action: str):
        """Handle action commands from the web viewer."""
        if action == 'reset' and self.avoidance is not None:
            self.avoidance.reset()
            print("[YOLO Viewer] Avoidance system reset")
        elif action == 'toggle_manual':
            self.manual_mode = not self.manual_mode
            if not self.manual_mode:
                # Returning to avoidance: clear inputs and reset state machine
                self.manual_steering = 0.0
                self.manual_throttle = 0.0
                if self.avoidance is not None:
                    self.avoidance.reset()
                    # Clear any stale STOP obstacle message left by manual mode
                    self.avoidance.clear_obstacle_override()
            mode_str = 'MANUAL' if self.manual_mode else 'AVOIDANCE'
            print(f"[YOLO Viewer] Control mode: {mode_str}")
        else:
            print(f"[YOLO Viewer] Unknown action: {action}")

    def _handle_manual_control(self, steering: float, throttle: float):
        """Update manual steering/throttle (only applied when manual_mode is True)."""
        if self.manual_mode:
            self.manual_steering = max(-1.0, min(1.0, float(steering)))
            self.manual_throttle = max(-1.0, min(1.0, float(throttle)))

    def _get_config(self) -> dict:
        """Return current parameter values for HTML template and /config endpoint."""
        cfg = {}
        if self.avoidance is not None:
            for name in AVOIDANCE_PARAMS:
                cfg[name] = getattr(self.avoidance, name, '')
        if self.detector is not None:
            for name in YOLO_PARAMS:
                cfg[name] = getattr(self.detector, name, '')
        cfg['frame_width'] = getattr(self.avoidance, 'frame_width', 768) if self.avoidance else 768
        cfg['frame_height'] = getattr(self.avoidance, 'frame_height', 384) if self.avoidance else 384
        return cfg
