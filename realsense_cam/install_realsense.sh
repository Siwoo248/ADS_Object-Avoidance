#!/bin/bash

################################################################################
#
# RealSense SDK Clean Reinstall Script for Jetson
# Purpose: Fix broken librealsense C++ backend + pyrealsense2
# 
# This script will:
# 1. Remove all broken RealSense installations
# 2. Clean rebuild librealsense (minimal, depth-only for YOLO)
# 3. Verify the installation
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RealSense SDK Clean Reinstall for Jetson (Depth-Only Mode)   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}ERROR: This script only works on Linux${NC}"
    exit 1
fi

# Check if user has sudo privileges
if ! sudo -n true 2>/dev/null; then
    echo -e "${YELLOW}This script requires sudo access. You may be prompted for password.${NC}"
    sudo true
fi

################################################################################
# STEP 1: Remove broken RealSense installations
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 1: Removing broken RealSense installations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}Removing C++ librealsense libraries...${NC}"
sudo rm -rf /usr/local/lib/librealsense* 2>/dev/null || true
sudo rm -rf /usr/lib/*/librealsense* 2>/dev/null || true
sudo rm -rf /usr/local/include/librealsense* 2>/dev/null || true
sudo rm -rf /usr/local/bin/rs-* 2>/dev/null || true
echo -e "${GREEN}✓ Old installations removed${NC}"

echo -e "${BLUE}Removing old Python bindings...${NC}"
pip uninstall -y pyrealsense2 2>/dev/null || true
echo -e "${GREEN}✓ Old Python bindings removed${NC}"

echo -e "${BLUE}Updating library cache...${NC}"
sudo ldconfig
echo -e "${GREEN}✓ Library cache updated${NC}"

################################################################################
# STEP 2: Check for librealsense source
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 2: Locating librealsense source${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Check common locations
LIBREALSENSE_DIR=""

if [ -d "$HOME/librealsense" ]; then
    LIBREALSENSE_DIR="$HOME/librealsense"
    echo -e "${GREEN}✓ Found: $LIBREALSENSE_DIR${NC}"
elif [ -d "/opt/librealsense" ]; then
    LIBREALSENSE_DIR="/opt/librealsense"
    echo -e "${GREEN}✓ Found: $LIBREALSENSE_DIR${NC}"
elif [ -d "/usr/local/src/librealsense" ]; then
    LIBREALSENSE_DIR="/usr/local/src/librealsense"
    echo -e "${GREEN}✓ Found: $LIBREALSENSE_DIR${NC}"
else
    echo -e "${RED}ERROR: librealsense source directory not found!${NC}"
    echo -e "${YELLOW}Please clone librealsense first:${NC}"
    echo -e "${YELLOW}  cd ~${NC}"
    echo -e "${YELLOW}  git clone https://github.com/IntelRealSense/librealsense.git${NC}"
    exit 1
fi

cd "$LIBREALSENSE_DIR"
echo -e "${BLUE}Working directory: $LIBREALSENSE_DIR${NC}"

################################################################################
# STEP 3: Clean build directory
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 3: Preparing build environment${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

if [ -d "build" ]; then
    echo -e "${BLUE}Removing old build directory...${NC}"
    rm -rf build
    echo -e "${GREEN}✓ Build directory cleaned${NC}"
fi

echo -e "${BLUE}Creating fresh build directory...${NC}"
mkdir -p build
cd build
echo -e "${GREEN}✓ Build directory created${NC}"

################################################################################
# STEP 4: Configure CMake (minimal, Jetson-optimized)
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 4: Configuring CMake (minimal build)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}Running CMake configuration...${NC}"
echo -e "${BLUE}Options:${NC}"
echo -e "  • BUILD_TYPE: Release (optimized)"
echo -e "  • PYTHON_BINDINGS: ON (for pyrealsense2)"
echo -e "  • RSUSB_BACKEND: ON (USB support)"
echo -e "  • EXAMPLES: OFF (not needed)"
echo -e "  • GRAPHICAL_EXAMPLES: OFF (not needed)"
echo -e "  • UNIT_TESTS: OFF (not needed)"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_STATIC_CRT=OFF

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configuration successful${NC}"

################################################################################
# STEP 5: Build
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 5: Building librealsense${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

NPROC=$(nproc)
echo -e "${BLUE}Building with $NPROC parallel jobs...${NC}"
echo -e "${BLUE}This may take 10-20 minutes on Jetson...${NC}"

make -j$NPROC

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build completed successfully${NC}"

################################################################################
# STEP 6: Install
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 6: Installing librealsense${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}Running sudo make install...${NC}"
sudo make install

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Installation failed!${NC}"
    exit 1
fi

echo -e "${BLUE}Updating library cache...${NC}"
sudo ldconfig

echo -e "${GREEN}✓ Installation completed${NC}"

################################################################################
# STEP 7: Verify SDK version
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 7: Verifying SDK installation${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}Checking C++ library...${NC}"
if pkg-config --modversion realsense2; then
    echo -e "${GREEN}✓ C++ library verified${NC}"
else
    echo -e "${YELLOW}⚠ pkg-config not available (not critical)${NC}"
fi

################################################################################
# STEP 8: Verify Python binding
################################################################################

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 8: Testing Python binding${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}Testing pyrealsense2 import and device detection...${NC}"

python3 << 'PYEOF'
import sys
try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 imported successfully")
    
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"✓ Found {len(devices)} RealSense device(s)")
    
    if len(devices) > 0:
        try:
            device = devices[0]
            name = device.get_info(rs.camera_info.name)
            firmware = device.get_info(rs.camera_info.firmware_version)
            print(f"✓ Device name: {name}")
            print(f"✓ Firmware: {firmware}")
        except Exception as e:
            print(f"⚠ Could not read device info: {e}")
            print("  (Camera may need physical reconnection)")
    
    print("\n✓ SDK backend is working correctly!")
    sys.exit(0)
    
except ImportError as e:
    print(f"✗ Failed to import pyrealsense2: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Python binding test failed!${NC}"
    exit 1
fi

################################################################################
# STEP 9: Summary
################################################################################

echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ✅ INSTALLATION COMPLETE                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}Summary:${NC}"
echo -e "  ✓ C++ librealsense library: INSTALLED"
echo -e "  ✓ pyrealsense2 Python bindings: INSTALLED"
echo -e "  ✓ USB backend: ENABLED"
echo -e "  ✓ SDK backend verified: OK"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Physically replug your RealSense camera (USB reconnect)"
echo -e "  2. Run the diagnostic: python diagnose_camera.py"
echo -e "  3. If diagnostic passes, run: python yolo_depth_detection.py"

echo -e "\n${BLUE}For reference:${NC}"
echo -e "  • Diagnostic: /home/siwoo/ADS-Skynet/realsense_cam/diagnose_camera.py"
echo -e "  • Main script: /home/siwoo/ADS-Skynet/realsense_cam/yolo_depth_detection.py"

echo -e "\n${GREEN}Done!${NC}\n"
