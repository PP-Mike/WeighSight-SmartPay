#!/bin/bash
# 从 SSH 启动板子桌面上的 GUI 程序
# 优先 X11（PyQt5 兼容性最好），Wayland 作为兜底
# 用法: ./run_gui.sh [程序路径]

APP="${1:-/home/bianbu/aiscale_system_fin_pro7.py}"
UID_NUM=$(id -u)
export XDG_RUNTIME_DIR="/run/user/${UID_NUM}"

echo "=== 环境诊断 ==="
echo "用户            : $(whoami) (uid=$UID_NUM)"
echo "会话类型        : ${XDG_SESSION_TYPE:-未知}"
echo "当前 DISPLAY    : ${DISPLAY:-空}"
echo "X11 socket      : $(ls /tmp/.X11-unix/ 2>/dev/null | tr '\n' ' ')"
echo "wayland socket  : $(ls "${XDG_RUNTIME_DIR}"/wayland-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

# 从 Xorg 进程参数里找真正的 -auth 路径
XORG_AUTH=$(ps -eo args 2>/dev/null | grep -m1 '[X]org' | grep -o -- '-auth [^ ]*' | awk '{print $2}')
echo "Xorg -auth      : ${XORG_AUTH:-未找到}"
echo

launch_x11() {
    export QT_QPA_PLATFORM=xcb
    export QT_X11_NO_MITSHM=1
    export DISPLAY=:0

    local candidates=(
        "$XORG_AUTH"
        "$HOME/.Xauthority"
        "${XDG_RUNTIME_DIR}/gdm/Xauthority"
        "${XDG_RUNTIME_DIR}/lightdm/Xauthority"
        "${XDG_RUNTIME_DIR}/.Xauthority"
        "${XDG_RUNTIME_DIR}/.mutter-Xwaylandauth"
    )
    for f in "${candidates[@]}"; do
        if [ -n "$f" ] && [ -f "$f" ] && [ -r "$f" ]; then
            export XAUTHORITY="$f"
            echo "[X11] XAUTHORITY=$f"
            break
        fi
    done
    [ -z "$XAUTHORITY" ] && echo "[X11] 未找到可读的 Xauthority"

    # 连通性探测
    if command -v xdpyinfo >/dev/null 2>&1; then
        if xdpyinfo -display :0 >/dev/null 2>&1; then
            echo "[X11] 显示连接正常"
        else
            echo "[X11] 无法连接 :0"
            echo "      请在桌面终端执行一次:  xhost +local:"
            return 1
        fi
    fi
    return 0
}

launch_wayland() {
    export QT_QPA_PLATFORM=wayland
    export WAYLAND_DISPLAY=wayland-0
    echo "[Wayland] WAYLAND_DISPLAY=wayland-0"
}

# ---------- 优先 X11 ----------
if [ -S /tmp/.X11-unix/X0 ]; then
    if launch_x11; then
        echo "[启动] 程序: $APP"
        cd "$(dirname "$APP")" || exit 1
        exec python3 "$APP"
    fi
    echo "[X11] 失败，尝试 Wayland"
fi

# ---------- 兜底 Wayland ----------
if [ -S "${XDG_RUNTIME_DIR}/wayland-0" ]; then
    launch_wayland
    echo "[启动] 程序: $APP"
    cd "$(dirname "$APP")" || exit 1
    exec python3 "$APP"
fi

echo "[错误] 没有可用的显示服务"
echo "       请确认板子已进入桌面（不是登录界面）"
exit 1
