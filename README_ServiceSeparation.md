# 🏗️ Separated Service Architecture

## 📋 **المشكلة والحل**
تم إنشاء **4 خدمات منفصلة** تتيح لك **الاختيار الكامل** في استخدام Vector Store أو عدمه:

---

## 🎯 **الخدمات المتاحة**

### **1. Embedding Only Service** (Port 8009)
- **الوظيفة**: تحويل النص إلى vectors فقط
- **المدخل**: `{"text": "machine learning"}`
- **المخرج**: `{"embedding": [0.1, 0.2, ...], "dimension": 384}`
- **الملف**: `services/embedding_service/embedding_only.py`

### **2. Vector Store Service** (Port 8007)
- **الوظيفة**: البحث في vectors المحفوظة
- **المدخل**: `{"dataset": "argsme", "query_vector": [0.1, 0.2, ...], "top_k": 5}`
- **المخرج**: `{"results": [["doc1", 0.95], ["doc2", 0.89]]}`
- **الملف**: `services/vector_store_service/main.py`

### **3. Traditional Search Service** (Port 8004)
- **الوظيفة**: البحث التقليدي (النص → النتائج مباشرة)
- **المدخل**: `{"dataset": "argsme", "query": "machine learning"}`
- **المخرج**: `{"results": [["doc1", 0.95], ["doc2", 0.89]]}`
- **الملف**: `services/embedding_service/main.py`

### **4. Unified Search Service** (Port 8006)
- **الوظيفة**: خدمة موحدة مع خيار استخدام Vector Store
- **المدخل**: `{"dataset": "argsme", "query": "machine learning", "use_vector_store": true}`
- **المخرج**: نتائج البحث مع معلومات الأداء
- **الملف**: `services/unified_search_service/main.py`

---

## 🚀 **كيفية التشغيل**

### **تشغيل خدمة واحدة:**
```bash
# تشغيل خدمة Embedding فقط
cd services/embedding_service
python -m uvicorn embedding_only:app --host 0.0.0.0 --port 8009

# تشغيل خدمة Vector Store
cd services/vector_store_service
python -m uvicorn main:app --host 0.0.0.0 --port 8007

# تشغيل الخدمة الموحدة
cd services/unified_search_service
python -m uvicorn main:app --host 0.0.0.0 --port 8006
```

### **تشغيل جميع الخدمات:**
```bash
# في نوافذ منفصلة
start cmd /k "cd services/embedding_service && python -m uvicorn embedding_only:app --host 0.0.0.0 --port 8009"
start cmd /k "cd services/vector_store_service && python -m uvicorn main:app --host 0.0.0.0 --port 8007"
start cmd /k "cd services/unified_search_service && python -m uvicorn main:app --host 0.0.0.0 --port 8006"
```

---

## 🎯 **أمثلة الاستخدام**

### **1. استخدام Vector Store Pipeline (الأسرع)**
```python
import requests

# Step 1: الحصول على embedding
response = requests.post("http://localhost:8009/embed", 
                       json={"text": "machine learning algorithms"})
embedding = response.json()["embedding"]

# Step 2: البحث في vector store
search_response = requests.post("http://localhost:8007/search",
                              json={
                                  "dataset": "argsme",
                                  "query_vector": embedding,
                                  "top_k": 5
                              })
results = search_response.json()["results"]
```

### **2. استخدام Traditional Search**
```python
import requests

# البحث التقليدي مباشرة
response = requests.post("http://localhost:8004/search",
                       json={
                           "dataset": "argsme", 
                           "query": "machine learning algorithms",
                           "top_k": 5
                       })
results = response.json()["results"]
```

### **3. استخدام Unified Service مع الخيار**
```python
import requests

# مع Vector Store (أسرع)
response = requests.post("http://localhost:8006/search",
                       json={
                           "dataset": "argsme",
                           "query": "machine learning",
                           "use_vector_store": True,  # 🎯 هنا الخيار
                           "top_k": 5
                       })

# بدون Vector Store (traditional)
response = requests.post("http://localhost:8006/search", 
                       json={
                           "dataset": "argsme",
                           "query": "machine learning",
                           "use_vector_store": False,  # 🎯 هنا الخيار
                           "top_k": 5
                       })
```

---

## 🔥 **مقارنة الأداء**

### **Vector Store Pipeline:**
- ⚡ **السرعة**: ~50ms للبحث
- 💾 **الذاكرة**: عالية (يحتاج vector store)
- 🎯 **الدقة**: عالية جداً
- 📊 **الاستخدام**: البحث السريع

### **Traditional Search:**
- ⏱️ **السرعة**: ~2000ms للبحث
- 💾 **الذاكرة**: منخفضة
- 🎯 **الدقة**: جيدة
- 📊 **الاستخدام**: البحث التقليدي

### **مقارنة تلقائية:**
```bash
curl "http://localhost:8006/compare/argsme?query=machine%20learning"
```

---

## 🔧 **التحكم في الخدمات**

### **Health Check:**
```bash
# فحص جميع الخدمات
curl http://localhost:8009/health  # Embedding
curl http://localhost:8007/health  # Vector Store  
curl http://localhost:8006/health  # Unified (يفحص جميع الخدمات)
```

### **معلومات الخدمة:**
```bash
curl http://localhost:8009/  # معلومات Embedding
curl http://localhost:8007/  # معلومات Vector Store
curl http://localhost:8006/  # معلومات Unified
```

---

## 🎯 **المزايا**

### **1. المرونة الكاملة:**
- اختيار بين Vector Store أو Traditional Search
- تشغيل الخدمات بشكل منفصل
- تحكم كامل في pipeline

### **2. الأداء الأمثل:**
- Vector Store للسرعة القصوى
- Traditional Search للموارد المحدودة
- مقارنة تلقائية للأداء

### **3. قابلية التوسع:**
- خدمات منفصلة تماماً
- إمكانية التوسع الأفقي
- إدارة موارد مستقلة

---

## 🏆 **الخلاصة**

الآن لديك **4 خدمات منفصلة** تتيح لك:

1. **الاختيار الكامل**: استخدام Vector Store أو عدمه
2. **المرونة**: تشغيل الخدمات بشكل منفصل
3. **الأداء**: مقارنة الطرق المختلفة
4. **التحكم**: تحكم كامل في pipeline

**الاستخدام المثالي:**
- للسرعة: استخدم Vector Store Pipeline
- للموارد المحدودة: استخدم Traditional Search  
- للتحكم الكامل: استخدم Unified Service مع خيار `use_vector_store` 