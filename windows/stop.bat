@echo off
:: Stop TensorFold server
:: by Screwed Up Tech — screwedup.tech

echo [*] Stopping TensorFold...
for /f "tokens=2" %%P in ('wmic process where "commandline like '%%tensorfold serve%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo [*] Killing PID %%P
    taskkill /F /PID %%P >nul 2>&1
)
echo [*] Done.
