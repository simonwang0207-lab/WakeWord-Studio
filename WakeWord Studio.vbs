Option Explicit
Dim shell, fso, root, pythonw, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonw) Then
  pythonw = root & "\.envs\livekit\Scripts\pythonw.exe"
End If
If Not fso.FileExists(pythonw) Then
  MsgBox "未找到 .venv 或本机开发环境 .envs\livekit，请先按 README 安装依赖。", 16, "WakeWord Studio"
  WScript.Quit 1
End If
command = Chr(34) & pythonw & Chr(34) & " " & Chr(34) & root & "\run_studio_modern.py" & Chr(34)
shell.Run command, 0, False
