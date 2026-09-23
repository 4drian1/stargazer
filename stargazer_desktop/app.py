"""Stargazer 桌面版入口。
启动 Flask 后端 + 定时巡检线程，然后开桌面窗口（pywebview）。
没有 pywebview（或未装桌面环境）时退回系统浏览器。
"""
import sys
import time
import urllib.request

import server


def wait_ready(port, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    port = server.start()
    url = f"http://127.0.0.1:{port}/"
    if not wait_ready(port):
        print("后端启动超时，请查看日志")
        sys.exit(1)
    print(f"Stargazer 后端已就绪: {url}")
    try:
        import webview
        webview.create_window("Stargazer", url, width=1180, height=780,
                              min_size=(900, 600))
        webview.start()
    except ImportError:
        import webbrowser
        print("未安装 pywebview，改用系统浏览器打开（pip install pywebview 可获得独立窗口）")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
