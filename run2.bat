set PYTHONIOENCODING=utf-8
set PYTHON=python
set APIKEY=sk-CDTYxdG8vL3Qd0IqU65r
set APIKEY2=T3BlbkFJzlBiaczs6ejYKP3JGnrY
set CHARNAME=paimon
set MODEL=gpt-3.5-turbo
set PORT=38438
%PYTHON% SocketServer2.py --APIKey %APIKEY%%APIKEY2% --model %MODEL% --character %CHARNAME% --port %PORT% --chatVer 3 --brainwash False --stream True 
