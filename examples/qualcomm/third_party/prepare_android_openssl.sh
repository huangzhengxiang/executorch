#!/bin/bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <install-dir> <ndk-root> <android-abi> <api-level>" >&2
  exit 1
fi

INSTALL_DIR="$1"
NDK_ROOT="$2"
ANDROID_ABI="$3"
API_LEVEL="$4"

OPENSSL_VERSION="3.0.13"
OPENSSL_ARCHIVE="openssl-${OPENSSL_VERSION}.tar.gz"
OPENSSL_URL="https://www.openssl.org/source/${OPENSSL_ARCHIVE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/openssl-src"
SRC_DIR="${WORK_DIR}/openssl-${OPENSSL_VERSION}"
ARCHIVE_PATH="${WORK_DIR}/${OPENSSL_ARCHIVE}"

mkdir -p "${WORK_DIR}"
mkdir -p "${INSTALL_DIR}"

if [ -f "${INSTALL_DIR}/lib/libssl.a" ] && [ -f "${INSTALL_DIR}/lib/libcrypto.a" ] && \
   [ -f "${INSTALL_DIR}/include/openssl/ssl.h" ]; then
  echo "OpenSSL already prepared at ${INSTALL_DIR}"
  exit 0
fi

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "Downloading OpenSSL ${OPENSSL_VERSION}..."
  curl -fL -o "${ARCHIVE_PATH}" "${OPENSSL_URL}"
fi

if [ ! -d "${SRC_DIR}" ]; then
  echo "Extracting OpenSSL ${OPENSSL_VERSION}..."
  tar -xzf "${ARCHIVE_PATH}" -C "${WORK_DIR}"
fi

case "${ANDROID_ABI}" in
  arm64-v8a)
    OPENSSL_TARGET="android-arm64"
    ;;
  armeabi-v7a)
    OPENSSL_TARGET="android-arm"
    ;;
  x86_64)
    OPENSSL_TARGET="android-x86_64"
    ;;
  x86)
    OPENSSL_TARGET="android-x86"
    ;;
  *)
    echo "Unsupported Android ABI: ${ANDROID_ABI}" >&2
    exit 1
    ;;
esac

export ANDROID_NDK_ROOT="${NDK_ROOT}"
export PATH="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}"

echo "Building OpenSSL ${OPENSSL_VERSION} for ${ANDROID_ABI} (API ${API_LEVEL})..."
cd "${SRC_DIR}"
make distclean >/dev/null 2>&1 || true
perl ./Configure "${OPENSSL_TARGET}" \
  -D__ANDROID_API__="${API_LEVEL}" \
  no-shared \
  no-tests \
  no-unit-test \
  no-engine \
  no-async \
  --prefix="${INSTALL_DIR}"
make -j"$(nproc)"
make install_sw
