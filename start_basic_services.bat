@echo off
echo 🚀 Starting Basic IR System Services...
echo ====================================

echo Starting basic services for IR System...
echo.

echo Terminal 1 - Preprocessing Service (Port 8002)
start "Preprocessing" cmd /k "uvicorn services.preprocessing_service.main:app --host 0.0.0.0 --port 8002"

timeout /t 2 /nobreak > nul

echo Terminal 2 - TF-IDF Service (Port 8003)
start "TF-IDF" cmd /k "uvicorn services.tfidf_service.main:app --host 0.0.0.0 --port 8003"

timeout /t 2 /nobreak > nul

echo Terminal 3 - Embedding Service (Port 8004)
start "Embedding" cmd /k "uvicorn services.embedding_service.main:app --host 0.0.0.0 --port 8004"

timeout /t 2 /nobreak > nul

echo Terminal 4 - Hybrid Service (Port 8005)
start "Hybrid" cmd /k "uvicorn services.hybrid_service.main:app --host 0.0.0.0 --port 8005"

timeout /t 5 /nobreak > nul

echo Terminal 5 - API Gateway (Port 8000)
start "API Gateway" cmd /k "uvicorn services.api_gateway.main:app --host 0.0.0.0 --port 8000"

echo.
echo ✅ Basic services started successfully!
echo.
echo Core Services:
echo   • Preprocessing: http://localhost:8002
echo   • TF-IDF: http://localhost:8003
echo   • Embedding: http://localhost:8004
echo   • Hybrid: http://localhost:8005
echo   • API Gateway: http://localhost:8000
echo.
echo 🌐 Access Interface: http://localhost:8000
echo.
echo ℹ️  Note: Optional services (Topic Detection, Query Suggestions, Vector Store)
echo    can be enabled from the interface once those services are running.
echo.
echo Press any key to close this window...
pause > nul 