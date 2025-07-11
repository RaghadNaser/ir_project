# Hybrid Search Model

نظام بحث هجين يجمع بين TF-IDF و Embedding models لتحسين دقة البحث.

## المميزات

- **Parallel Fusion**: دمج نتائج TF-IDF و Embedding في مرحلة واحدة
- **Serial Fusion**: استخدام نموذج واحد للمرشحين ثم إعادة الترتيب بالنموذج الآخر
- **قابلية التخصيص**: إمكانية تعديل أوزان النماذج
- **دعم متعدد الداتاست**: يعمل مع argsme و wikir
- **تحسين الأداء**: استخدام المرشحين لتقليل الوقت

## البنية

```
models/
├── hybrid_model.py      # النموذج الهجين الرئيسي
├── tfidf_model.py       # نموذج TF-IDF
├── embedding_model.py   # نموذج Embedding
└── inverted_index.py    # الفهرس المعكوس

config/
├── settings.py          # إعدادات التطبيق
└── logging_config.py    # إعدادات التسجيل

utils/
└── path_utils.py        # إدارة المسارات
```

## طرق البحث

### 1. Parallel Fusion
```python
results = model.search(query, method="parallel", top_k=10)
```

### 2. Serial TF-IDF First
```python
results = model.search(
    query, 
    method="serial_tfidf_first", 
    top_k=10, 
    first_stage_k=2000
)
```

### 3. Serial Embedding First
```python
results = model.search(
    query, 
    method="serial_embedding_first", 
    top_k=10, 
    first_stage_k=2000
)
```

## الاستخدام

```python
from models.hybrid_model import HybridModel

# إنشاء النموذج
model = HybridModel(
    dataset_name="argsme",
    tfidf_weight=0.4,
    embedding_weight=0.6
)

# البحث
results = model.search(
    query="artificial intelligence",
    method="parallel",
    top_k=10
)

# عرض النتائج
for doc_id, score, individual_scores in results:
    print(f"Doc ID: {doc_id}, Score: {score:.4f}")
    print(f"  TF-IDF: {individual_scores['tfidf']:.4f}")
    print(f"  Embedding: {individual_scores['embedding']:.4f}")
```

## المعاملات

### HybridModel
- `dataset_name`: اسم الداتاست ("argsme", "wikir")
- `tfidf_weight`: وزن TF-IDF (افتراضي: 0.4)
- `embedding_weight`: وزن Embedding (افتراضي: 0.6)

### search()
- `query`: نص البحث
- `method`: طريقة الدمج ("parallel", "serial_tfidf_first", "serial_embedding_first")
- `top_k`: عدد النتائج المطلوبة
- `first_stage_k`: عدد المرشحين في المرحلة الأولى (للطرق التسلسلية)

## النتائج

كل نتيجة تحتوي على:
- `doc_id`: معرف المستند
- `score`: الدرجة النهائية المجمعة
- `individual_scores`: الدرجات الفردية لكل نموذج

## التشغيل

```bash
# اختبار النموذج
python test_hybrid_model.py
```

## المتطلبات

- Python 3.8+
- numpy
- scikit-learn
- sentence-transformers
- joblib
- scipy

## ملاحظات

1. تأكد من وجود ملفات النماذج المدربة في المسارات الصحيحة
2. النموذج يحمل الملفات عند أول استخدام
3. يمكن تخصيص أوزان النماذج حسب احتياجات التطبيق
4. الطرق التسلسلية أسرع من Parallel للاستعلامات الكبيرة 