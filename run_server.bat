set PYTHONIOENCODING=utf-8
set PYTHON=python
set CHARNAME=paimon
set MODEL=gpt-3.5-turbo
set PORT=38438
%PYTHON% SocketServer.py --model %MODEL% --character %CHARNAME% --port %PORT%
