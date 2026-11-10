#!/usr/bin/env bash
# Idempotent toolchain for assignment2 and assignment3.
# The default branch does not contain those trees, so this script does not
# depend on them. It also does not run apt-get: archive.ubuntu.com stalls
# inside environment builds, and the default image already has Python 3.12.
set -euo pipefail

PY=/usr/bin/python3
if [[ ! -x "${PY}" ]]; then
  echo "expected ${PY} on the Cloud Agent base image" >&2
  exit 1
fi

# Do not upgrade Debian's pip: it has no RECORD file and pip refuses to uninstall it.
# --ignore-installed: Debian's numpy has no RECORD file, so pip cannot uninstall it.
sudo "${PY}" -m pip install --break-system-packages --ignore-installed \
  numpy \
  scipy \
  matplotlib \
  pillow \
  opencv-python-headless \
  python-docx \
  reportlab \
  pypdf

sudo mkdir -p /usr/local/lib/python3.12/dist-packages
sudo tee /usr/local/lib/python3.12/dist-packages/sitecustomize.py >/dev/null << 'EOF'
import os
os.environ.setdefault("MPLBACKEND", "Agg")
EOF

# Regular files, not symlinks. tee follows a symlink and would overwrite
# the target; Python also ignores a venv when its binary is reached by a
# symlink outside that venv.
install_wrapper() {
  local name="$1"
  local target="$2"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" << EOF
#!/bin/sh
exec ${target} "\$@"
EOF
  sudo mv -f "${tmp}" "/usr/local/bin/${name}"
  sudo chmod 755 "/usr/local/bin/${name}"
}
install_wrapper python /usr/bin/python3
install_wrapper python3 /usr/bin/python3
install_wrapper pip /usr/bin/pip3
install_wrapper pip3 /usr/bin/pip3

timeout 60 /usr/local/bin/python -c 'import matplotlib, numpy, scipy, PIL, cv2, docx, reportlab, pypdf; assert matplotlib.get_backend().lower() == "agg"'
timeout 30 /usr/local/bin/pip --version
