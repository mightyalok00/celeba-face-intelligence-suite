@echo off
setlocal
set "MYSQL_BIN=D:\MySQL\bin\mysqld.exe"
set "DATA_DIR=D:\scan\mysql_data"

tasklist /FI "IMAGENAME eq mysqld.exe" 2>NUL | find /I "mysqld.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] MySQL Server is already running on 127.0.0.1:3306 - celeba_db
    exit /b 0
)

echo [INFO] Starting MySQL Server with datadir "%DATA_DIR%"...
start "MySQL Server" /B "%MYSQL_BIN%" --datadir="%DATA_DIR%" --console

timeout /t 3 >NUL
tasklist /FI "IMAGENAME eq mysqld.exe" 2>NUL | find /I "mysqld.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo [SUCCESS] MySQL Server is running on 127.0.0.1:3306 - celeba_db
) else (
    echo [ERROR] Failed to start MySQL Server. Please verify "%MYSQL_BIN%".
)
