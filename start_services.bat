@echo off
echo 🚀 Starting IR System Services...
echo ====================================

echo Starting services on specified ports...
echo.

echo Terminal 1 - Preprocessing Service (Port 8002)
start "Preprocessing Service" cmd /k "uvicorn services.preprocessing_service.main:app --host 0.0.0.0 --port 8002"

timeout /t 2 /nobreak > nul

echo Terminal 2 - TF-IDF Service (Port 8003)
start "TF-IDF Service" cmd /k "uvicorn services.tfidf_service.main:app --host 0.0.0.0 --port 8003"

timeout /t 2 /nobreak > nul

echo Terminal 3 - Embedding Service (Port 8004)
start "Embedding Service" cmd /k "uvicorn services.embedding_service.main:app --host 0.0.0.0 --port 8004"

timeout /t 2 /nobreak > nul

echo Terminal 4 - Hybrid Service (Port 8005)
start "Hybrid Service" cmd /k "uvicorn services.hybrid_service.main:app --host 0.0.0.0 --port 8005"

timeout /t 2 /nobreak > nul

echo Terminal 5 - Topic Detection Service (Port 8006)
start "Topic Detection Service" cmd /k "uvicorn services.topic_detection_service.main:app --host 0.0.0.0 --port 8006"

timeout /t 2 /nobreak > nul

echo Terminal 6 - Vector Store Service (Port 8008)
start "Vector Store Service" cmd /k "uvicorn services.vector_store_service.main:app --host 0.0.0.0 --port 8008"

timeout /t 2 /nobreak > nul

echo Terminal 7 - Query Suggestion Service (Port 8010)
start "Query Suggestion Service" cmd /k "uvicorn services.query_suggestion_service.main:app --host 0.0.0.0 --port 8010"

timeout /t 5 /nobreak > nul

echo Terminal 8 - API Gateway (Port 8000)
start "API Gateway" cmd /k "uvicorn services.api_gateway.main:app --host 0.0.0.0 --port 8000"

echo.
echo ✅ All services are starting in separate terminals...
echo.
echo Services:
echo   • Preprocessing Service: http://localhost:8002
echo   • TF-IDF Service: http://localhost:8003
echo   • Embedding Service: http://localhost:8004
echo   • Hybrid Service: http://localhost:8005
echo   • Topic Detection Service: http://localhost:8006
echo   • Vector Store Service: http://localhost:8008
echo   • Query Suggestion Service: http://localhost:8010
echo   • API Gateway: http://localhost:8000
echo.
echo 🌐 Main Interface: http://localhost:8000
echo.
echo Press any key to close this window...
pause > nul 