# دليل استخدام APIs خدمة الكشف عن المواضيع في Postman

## 📋 نظرة عامة
خدمة الكشف عن المواضيع (Topic Detection Service) تعمل على **البورت 8006** وتوفر مجموعة شاملة من APIs لاكتشاف المواضيع في النصوص باستخدام نماذج مدربة على بيانات ARGSME و WikiIR.

## 🚀 كيفية استيراد المجموعة في Postman

### الطريقة الأولى: استيراد الملف
1. افتح Postman
2. اضغط على "Import"
3. اختر "File" 
4. حدد الملف `topic_detection_postman_collection.json`
5. اضغط "Import"

### الطريقة الثانية: استيراد من Raw JSON
1. افتح Postman
2. اضغط على "Import"
3. اختر "Raw text"
4. انسخ محتوى الملف JSON والصقه
5. اضغط "Continue" ثم "Import"

## 🏗️ بنية المجموعة

### 1. **Service Information** - معلومات الخدمة
- **Root - Service Info**: معلومات أساسية عن الخدمة
- **Health Check**: فحص حالة الخدمة
- **Model Info - ARGSME**: معلومات نموذج ARGSME
- **Model Info - WikiIR**: معلومات نموذج WikiIR
- **Available Datasets**: قائمة بالبيانات المتاحة

### 2. **Topic Detection** - الكشف عن المواضيع
- **Detect Topics - ARGSME (Basic)**: كشف أساسي للمواضيع العربية
- **Detect Topics - WikiIR (Basic)**: كشف أساسي للمواضيع الإنجليزية
- **Detect Topics - Advanced Settings**: إعدادات متقدمة
- **Detect Topics - Using 'text' field**: استخدام حقل النص البديل

### 3. **Topic Prediction (Alias)** - التنبؤ بالمواضيع
- **Predict Topics - ARGSME**: تنبؤ للبيانات العربية
- **Predict Topics - WikiIR**: تنبؤ للبيانات الإنجليزية

### 4. **Topic Suggestions** - اقتراحات المواضيع
- **Suggest Topics - ARGSME (Default)**: اقتراحات افتراضية
- **Suggest Topics - WikiIR (Default)**: اقتراحات افتراضية
- **Suggest Topics - Custom Limit**: اقتراحات بحد مخصص

### 5. **Get Topics** - الحصول على المواضيع
- **Get Topics - ARGSME (Default 50)**: أول 50 موضوع
- **Get Topics - WikiIR (Default 50)**: أول 50 موضوع
- **Get Topics - Custom Limit (100)**: حد مخصص 100
- **Get Topics - Small Limit (10)**: حد صغير 10

### 6. **Error Testing** - اختبار الأخطاء
- **Invalid Dataset Test**: اختبار بيانات غير صحيحة
- **Empty Query Test**: اختبار استعلام فارغ
- **Missing Query Test**: اختبار استعلام مفقود

### 7. **Performance Testing** - اختبار الأداء
- **Long Query Test**: اختبار استعلام طويل
- **High Precision Test**: اختبار دقة عالية
- **Max Topics Test**: اختبار أقصى عدد مواضيع

## 🔧 تشغيل الخدمة

### 1. تشغيل الخدمة محلياً
```bash
cd services/topic_detection_service
python main.py
```

### 2. التحقق من تشغيل الخدمة
```bash
curl http://localhost:8006/health
```

## 📊 الـ APIs المتاحة

### 1. **GET /** - معلومات الخدمة
- **الوصف**: معلومات أساسية عن الخدمة وقائمة بجميع الـ endpoints
- **المثال**:
```json
{
  "service": "Topic Detection Service",
  "version": "1.0.0",
  "supported_datasets": ["argsme", "wikir"],
  "endpoints": {...}
}
```

### 2. **POST /detect-topics** - كشف المواضيع
- **الوصف**: يكشف المواضيع في النص المدخل
- **المعاملات**:
  - `query` أو `text`: النص المراد تحليله
  - `dataset`: "argsme" أو "wikir"
  - `max_topics`: أقصى عدد مواضيع (افتراضي: 10)
  - `min_relevance_score`: أقل درجة صلة (افتراضي: 0.1)

**مثال للطلب**:
```json
{
  "query": "الذكاء الاصطناعي والتعلم الآلي",
  "dataset": "argsme",
  "max_topics": 10,
  "min_relevance_score": 0.1
}
```

**مثال للاستجابة**:
```json
{
  "query": "الذكاء الاصطناعي والتعلم الآلي",
  "dataset": "argsme",
  "detected_topics": [
    {
      "topic": "الذكاء الاصطناعي",
      "relevance_score": 0.95,
      "frequency": 150,
      "coverage_ratio": 0.08,
      "doc_count": 45
    }
  ],
  "similar_topics": [...],
  "processing_time": 0.123
}
```

### 3. **POST /predict** - التنبؤ بالمواضيع
- **الوصف**: نفس `/detect-topics` لكن بتنسيق مبسط
- **الاستجابة**:
```json
{
  "topics": [
    {
      "topic": "الذكاء الاصطناعي",
      "score": 0.95,
      "frequency": 150
    }
  ],
  "dataset": "argsme",
  "query": "الذكاء الاصطناعي",
  "total_topics": 1
}
```

### 4. **GET /suggest-topics** - اقتراح المواضيع
- **الوصف**: يقترح مواضيع شائعة من البيانات
- **المعاملات**:
  - `dataset`: "argsme" أو "wikir"
  - `limit`: عدد الاقتراحات (افتراضي: 20)

### 5. **GET /topics** - الحصول على المواضيع
- **الوصف**: يعرض قائمة بالمواضيع المتاحة
- **المعاملات**:
  - `dataset`: "argsme" أو "wikir"
  - `limit`: عدد المواضيع (افتراضي: 50)

### 6. **GET /health** - فحص الحالة
- **الوصف**: يتحقق من حالة الخدمة والنماذج المحملة
- **الاستجابة**:
```json
{
  "status": "healthy",
  "models": {
    "argsme": "loaded",
    "wikir": "loaded"
  },
  "service": "Topic Detection Service"
}
```

### 7. **GET /model-info** - معلومات النموذج
- **الوصف**: معلومات مفصلة عن النموذج المحدد
- **المعاملات**:
  - `dataset`: "argsme" أو "wikir"

### 8. **GET /datasets** - البيانات المتاحة
- **الوصف**: قائمة بجميع البيانات المتاحة وحالتها

## 🎯 سيناريوهات الاستخدام

### 1. **الاستخدام الأساسي**
1. ابدأ بـ **Health Check** للتأكد من تشغيل الخدمة
2. استخدم **Root - Service Info** لفهم الخدمة
3. جرب **Detect Topics - ARGSME (Basic)** للنصوص العربية
4. جرب **Detect Topics - WikiIR (Basic)** للنصوص الإنجليزية

### 2. **الاستخدام المتقدم**
1. استخدم **Model Info** لفهم النماذج المتاحة
2. جرب **Advanced Settings** مع معاملات مختلفة
3. استخدم **Suggest Topics** للحصول على اقتراحات
4. جرب **Get Topics** لاستكشاف المواضيع المتاحة

### 3. **اختبار الأداء**
1. جرب **Long Query Test** للنصوص الطويلة
2. استخدم **High Precision Test** للدقة العالية
3. جرب **Max Topics Test** لأقصى عدد مواضيع

### 4. **اختبار الأخطاء**
1. جرب **Invalid Dataset Test** للتحقق من معالجة الأخطاء
2. استخدم **Empty Query Test** للاستعلامات الفارغة
3. جرب **Missing Query Test** للمعاملات المفقودة

## 📈 نصائح للاستخدام الأمثل

### 1. **اختيار البيانات المناسبة**
- استخدم **"argsme"** للنصوص العربية والجدالية
- استخدم **"wikir"** للنصوص الإنجليزية والمعلوماتية

### 2. **ضبط المعاملات**
- `max_topics`: ابدأ بـ 10 ثم زد حسب الحاجة
- `min_relevance_score`: 0.1 للعموم، 0.5+ للدقة العالية

### 3. **مراقبة الأداء**
- راقب `processing_time` في الاستجابات
- استخدم اختبارات الأداء للنصوص الطويلة

### 4. **معالجة الأخطاء**
- تحقق من `status_code` في الاستجابات
- استخدم اختبارات الأخطاء لفهم السلوك المتوقع

## 🔍 متغيرات Postman

المجموعة تحتوي على متغيرات مفيدة:
- `base_url`: http://localhost:8006
- `service_name`: Topic Detection Service

## 🧪 اختبارات تلقائية

كل طلب يحتوي على اختبارات تلقائية:
- فحص زمن الاستجابة (أقل من 5 ثوان)
- فحص تنسيق JSON
- فحص حالة النجاح (200, 201, 202)

## 🚨 استكشاف الأخطاء

### مشاكل شائعة:
1. **Connection refused**: تأكد من تشغيل الخدمة على البورت 8006
2. **Model not loaded**: تحقق من وجود ملفات النماذج
3. **Invalid dataset**: استخدم "argsme" أو "wikir" فقط
4. **Empty query**: تأكد من إدخال نص في `query` أو `text`

### حلول:
1. تشغيل الخدمة: `python services/topic_detection_service/main.py`
2. فحص الحالة: استخدم `/health` endpoint
3. فحص النماذج: استخدم `/model-info` endpoint
4. فحص البيانات: استخدم `/datasets` endpoint

## 📚 مثال شامل

```bash
# 1. تشغيل الخدمة
cd services/topic_detection_service
python main.py

# 2. فحص الحالة
curl http://localhost:8006/health

# 3. كشف المواضيع
curl -X POST http://localhost:8006/detect-topics \
  -H "Content-Type: application/json" \
  -d '{
    "query": "الذكاء الاصطناعي في التعليم",
    "dataset": "argsme",
    "max_topics": 5
  }'
```

## 🔗 روابط مفيدة

- **الخدمة الأساسية**: http://localhost:8006/
- **فحص الحالة**: http://localhost:8006/health
- **الوثائق التلقائية**: http://localhost:8006/docs (FastAPI Swagger)
- **الوثائق البديلة**: http://localhost:8006/redoc (ReDoc)

---

هذا الدليل يوفر كل ما تحتاجه لاستخدام خدمة الكشف عن المواضيع بفعالية في Postman! 🚀 