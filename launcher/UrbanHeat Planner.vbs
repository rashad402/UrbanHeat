' Starts the UrbanHeat Planner with no console window, not even a flash.
'
'   UrbanHeat Planner.vbs          start the planner (or just open it if it is already running)
'   UrbanHeat Planner.vbs stop     shut a running planner down
'
' A PowerShell window cannot be hidden early enough from a normal shortcut and flashes for a moment.
' Run with window style 0 from here, it never appears. The real work is in planner_tray.ps1.

Dim shell, fso, here, action
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)

action = "start"
If WScript.Arguments.Count > 0 Then
    If LCase(WScript.Arguments(0)) = "stop" Then action = "stop"
End If

shell.Run "powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & _
          here & "\planner_tray.ps1"" -Action " & action, 0, False
