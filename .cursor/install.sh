#!/usr/bin/env bash
# Idempotent toolchain for the computer-vision homework on assignment2 and assignment3.
# The default branch does not vendor those trees, so this script does not depend on them.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Ubuntu mirrors occasionally return 502 during Cloud Agent installs.
apt_update() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if sudo apt-get update; then
      return 0
    fi
    echo "apt-get update failed (attempt ${attempt}); retrying in $((attempt * 10))s" >&2
    sleep $((attempt * 10))
  done
  return 1
}
apt_update

sudo apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  poppler-utils \
  libreoffice-writer \
  fonts-liberation \
  fonts-dejavu-core \
  fonts-wqy-microhei \
  fonts-noto-cjk

VENV=/opt/cv-homework
if [[ ! -x "${VENV}/bin/python" ]]; then
  sudo mkdir -p "${VENV}"
  sudo chown "$(id -u):$(id -g)" "${VENV}"
  python3 -m venv "${VENV}"
fi

"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install \
  numpy \
  scipy \
  matplotlib \
  pillow \
  opencv-python-headless \
  python-docx \
  reportlab \
  pypdf

SITE="$("${VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cat > "${SITE}/sitecustomize.py" << 'EOF'
import os
os.environ.setdefault("MPLBACKEND", "Agg")
EOF

# Wrappers, not symlinks. Python locates a venv from the executable path,
# and a symlink under /usr/local/bin resolves to system Python.
# Write via a temp file and mv so we never follow an existing symlink into the venv.
for cmd in python python3 pip pip3; do
  tmp="$(mktemp)"
  cat > "${tmp}" << EOF
#!/bin/sh
exec ${VENV}/bin/${cmd} "\$@"
EOF
  sudo mv -f "${tmp}" "/usr/local/bin/${cmd}"
  sudo chmod 755 "/usr/local/bin/${cmd}"
done

timeout 60 /usr/local/bin/python -c 'import numpy, scipy, matplotlib, PIL, cv2, docx, reportlab, pypdf'
timeout 30 /usr/local/bin/pip --version
timeout 60 soffice --version >/dev/null
