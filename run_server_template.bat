set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\jack\anaconda3\Scripts\conda.exe run -n dlife --no-capture-output --live-stream python
set APIKEY=<CHAPGPT_API_KEY>
set CHARNAME=paimon
set MODEL=gpt-3.5-turbo
set PORT=38438
%PYTHON% SocketServer.py --APIKey %APIKEY% --model %MODEL% --character %CHARNAME% --port %PORT% --stream True 
