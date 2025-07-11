# نظام استرجاع المعلومات المتكامل

## نظرة عامة

هذا النظام يجمع بين النماذج التقليدية (TF-IDF) والنماذج الحديثة (Embedding) لتحسين دقة البحث في استرجاع المعلومات.

## البنية الجديدة

```
Final_project_IR/
├── config/                     # إعدادات التطبيق
│   ├── __init__.py
│   ├── logging_config.py      # إعدادات التسجيل
│   └── settings.py            # إعدادات عامة
├── models/                     # النماذج الجديدة
│   ├── __init__.py
│   ├── hybrid_model.py        # النموذج الهجين الرئيسي
│   ├── tfidf_model.py         # نموذج TF-IDF محسن
│   ├── embedding_model.py     # نموذج Embedding
│   └── inverted_index.py      # الفهرس المعكوس
├── utils/                      # الأدوات المساعدة
│   ├── __init__.py
│   └── path_utils.py          # إدارة المسارات
├── services/                   # الخدمات
│   ├── api_gateway/           # بوابة API مع واجهة ويب
│   ├── hybrid_service/        # الخدمة الهجينة (محدثة)
│   ├── tfidf_service/         # خدمة TF-IDF
│   ├── embedding_service/     # خدمة Embedding
│   ├── preprocessing_service/ # خدمة المعالجة المسبقة
│   └── indexing_service/      # خدمة الفهرسة
├── data/                       # البيانات
├── offline/                    # سكريبتات التدريب
├── start_services.py          # تشغيل جميع الخدمات
├── test_hybrid_model.py       # اختبار النموذج الهجين
└── README_Integration.md      # هذا الملف
```

## المميزات الجديدة

### 1. النموذج الهجين المحسن
- **ثلاث طرق دمج**:
  - `parallel`: دمج مباشر لنتائج النموذجين
  - `serial_tfidf_first`: TF-IDF للمرشحين، ثم Embedding لإعادة الترتيب
  - `serial_embedding_first`: Embedding للمرشحين، ثم TF-IDF لإعادة الترتيب

### 2. أوزان قابلة للتخصيص
- إمكانية تعديل أوزان TF-IDF و Embedding
- واجهة تفاعلية لضبط الأوزان

### 3. تحسين الأداء
- استخدام المرشحين لتقليل وقت المعالجة
- تحميل النماذج مرة واحدة فقط

### 4. واجهة مستخدم محسنة
- عرض الدرجات الفردية لكل نموذج
- قياس وقت التنفيذ
- خيارات متقدمة للبحث الهجين

## كيفية التشغيل

### 1. تشغيل جميع الخدمات دفعة واحدة

```bash
# تشغيل جميع الخدمات
python start_services.py
```

### 2. تشغيل الخدمات بشكل منفصل

```bash
# تشغيل API Gateway (الواجهة الرئيسية)
cd services/api_gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# تشغيل الخدمة الهجينة
cd services/hybrid_service
uvicorn main:app --host 0.0.0.0 --port 8005 --reload

# تشغيل خدمة TF-IDF
cd services/tfidf_service
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# تشغيل خدمة Embedding
cd services/embedding_service
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

### 3. الوصول للواجهة

افتح المتصفح واذهب إلى: `http://localhost:8000`

## استخدام API

### البحث الهجين

```bash
curl -X POST "http://localhost:8005/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "argsme",
    "query": "artificial intelligence",
    "top_k": 10,
    "method": "parallel",
    "tfidf_weight": 0.4,
    "embedding_weight": 0.6
  }'
```

### مقارنة الطرق

```bash
curl -X POST "http://localhost:8005/compare_methods" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "argsme",
    "query": "artificial intelligence",
    "top_k": 5
  }'
```

### معلومات النموذج

```bash
curl "http://localhost:8005/model_info/argsme"
```

## استخدام النموذج مباشرة

```python
from models.hybrid_model import HybridModel

# إنشاء النموذج
model = HybridModel(
    dataset_name="argsme",
    tfidf_weight=0.4,
    embedding_weight=0.6
)

# البحث بالطريقة المتوازية
results = model.search(
    query="artificial intelligence",
    method="parallel",
    top_k=10
)

# البحث بالطريقة التسلسلية
results = model.search(
    query="artificial intelligence",
    method="serial_tfidf_first",
    top_k=10,
    first_stage_k=2000
)

# عرض النتائج
for doc_id, score, individual_scores in results:
    print(f"Doc ID: {doc_id}")
    print(f"  Final Score: {score:.4f}")
    print(f"  TF-IDF Score: {individual_scores['tfidf']:.4f}")
    print(f"  Embedding Score: {individual_scores['embedding']:.4f}")
```

## اختبار النظام

```bash
# اختبار النموذج الهجين
python test_hybrid_model.py
```

## المنافذ المستخدمة

| الخدمة | المنفذ | الوصف |
|--------|--------|-------|
| API Gateway | 8000 | الواجهة الرئيسية |
| Preprocessing | 8001 | معالجة النصوص |
| Indexing | 8002 | الفهرسة |
| TF-IDF | 8003 | البحث بـ TF-IDF |
| Embedding | 8004 | البحث بـ Embedding |
| Hybrid | 8005 | البحث الهجين |

## استكشاف الأخطاء

### 1. مشاكل تحميل النماذج
- تأكد من وجود ملفات النماذج في المسارات الصحيحة
- تحقق من صحة أسماء الملفات

### 2. مشاكل المنافذ
- استخدم `start_services.py` لإيقاف العمليات الموجودة
- أو استخدم `netstat -ano | findstr :8000` (Windows) أو `lsof -i :8000` (Linux)

### 3. مشاكل الذاكرة
- قلل `first_stage_k` للطرق التسلسلية
- استخدم `top_k` أصغر

## التطوير المستقبلي

1. **إضافة cache**: لتخزين النتائج المتكررة
2. **تحسين التوازي**: استخدام ThreadPoolExecutor
3. **إضافة metrics**: قياس دقة البحث
4. **دعم المزيد من الداتاست**: إضافة داتاست جديدة
5. **واجهة تحليلية**: رسوم بيانية للنتائج

## المساهمة

لإضافة ميزات جديدة أو تحسين النظام:

1. أضف النموذج الجديد في `models/`
2. حدث `config/settings.py` إذا لزم الأمر
3. أضف الخدمة في `services/`
4. حدث `start_services.py`
5. أضف الاختبارات في `test_hybrid_model.py`
6. حدث التوثيق 