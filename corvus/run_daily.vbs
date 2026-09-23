' 隐藏窗口运行 Corvus 每日全量任务（0=不显示窗口，False=不等待）
CreateObject("WScript.Shell").Run "cmd /c D:\stargazer\corvus\run_daily.bat", 0, False
