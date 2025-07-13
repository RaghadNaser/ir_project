# 🤖 Professional AI Agent Service

## نظرة عامة | Overview

نظام ذكي متقدم للبحث في المعلومات يدعم اللغتين العربية والإنجليزية مع خوارزميات بحث متعددة.

Advanced intelligent information retrieval system supporting both Arabic and English languages with multiple search algorithms.

## ✨ المميزات | Features

### 🌐 دعم متعدد اللغات | Multi-Language Support
- **العربية** - دعم كامل للبحث والرد باللغة العربية
- **English** - Full support for search and responses in English
- **Language Detection** - Automatic language detection
- **Bilingual Interface** - Interface supports both languages

### 🔍 طرق البحث المتعددة | Multiple Search Methods
- **Hybrid Search** - Combines TF-IDF and Embedding for best results
- **TF-IDF Search** - Keyword-based search
- **Embedding Search** - Semantic search using embeddings

### 🧠 الذكاء الاصطناعي المتقدم | Advanced AI
- **Context Awareness** - Understands conversation context
- **Query Refinement** - Automatically improves search queries
- **Intelligent Response Generation** - Generates natural language responses
- **Document Analysis** - Extracts and summarizes relevant content

## 🚀 التشغيل | Getting Started

### 1. تشغيل جميع الخدمات | Start All Services

```bash
python start_services_fixed.py
```

### 2. الوصول للواجهة | Access the Interface

- **الواجهة الرئيسية**: http://localhost:8000
- **واجهة الوكيل الذكي**: http://localhost:8000/agent
- **واجهة المحادثة**: http://localhost:8000/chat

### 3. اختبار النظام | Test the System

```bash
python test_agent_fixed.py
```

## 🎯 كيفية الاستخدام | How to Use

### اختيار اللغة | Language Selection
1. في واجهة الوكيل، ستجد أزرار اختيار اللغة في الأعلى
2. اختر "العربية" للبحث باللغة العربية
3. اختر "English" للبحث باللغة الإنجليزية

### اختيار طريقة البحث | Search Method Selection
1. اختر "Hybrid" للحصول على أفضل النتائج (مزيج من TF-IDF والـ Embedding)
2. اختر "TF-IDF" للبحث بالكلمات المفتاحية
3. اختر "Embedding" للبحث الدلالي

### أمثلة على الاستعلامات | Query Examples

#### العربية | Arabic
```
ابحث عن الذكاء الاصطناعي
ما هو التعلم الآلي؟
شرح مفاهيم الذكاء الاصطناعي
```

#### English
```
Explain machine learning concepts
Find information about climate change
Compare different search methods
What is artificial intelligence?
```

## 🔧 التكوين | Configuration

### منافذ الخدمات | Service Ports
- **API Gateway**: 8000
- **Preprocessing**: 8002
- **TF-IDF Service**: 8003
- **Embedding Service**: 8004
- **Hybrid Service**: 8005
- **Topic Detection**: 8006
- **Query Suggestions**: 8010
- **Agent Service**: 8011

### قاعدة البيانات | Database
- **Location**: `data/ir_database_combined.db`
- **Datasets**: ARGSME, WIKIR
- **Content**: Debate topics, arguments, and conclusions

## 📊 الاستجابة | Response Format

```json
{
    "response": "النص المولد باللغة المحددة",
    "documents": [
        {
            "id": "doc_id",
            "title": "عنوان المستند",
            "content": "محتوى المستند",
            "score": 0.95,
            "source": "argsme",
            "topic": "موضوع المستند"
        }
    ],
    "confidence": 0.8,
    "language": "ar",
    "search_method": "hybrid"
}
```

## 🛠️ استكشاف الأخطاء | Troubleshooting

### مشاكل شائعة | Common Issues

1. **الخدمة لا تستجيب | Service Not Responding**
   - تأكد من تشغيل جميع الخدمات
   - تحقق من المنافذ المفتوحة

2. **لا توجد نتائج | No Results Found**
   - جرب كلمات مفتاحية مختلفة
   - تأكد من اختيار مجموعة البيانات الصحيحة

3. **خطأ في الاتصال | Connection Error**
   - تحقق من تشغيل قاعدة البيانات
   - تأكد من وجود ملف `ir_database_combined.db`

### سجلات الأخطاء | Error Logs
- تحقق من سجلات الخدمات في Terminal
- استخدم `python test_agent_fixed.py` لاختبار الاتصال

## 🔄 التحديثات الأخيرة | Recent Updates

### v2.0 - Language and Search Method Support
- ✅ إضافة دعم اللغة العربية والإنجليزية
- ✅ إضافة أزرار اختيار اللغة في الواجهة
- ✅ إضافة اختيار طريقة البحث (Hybrid, TF-IDF, Embedding)
- ✅ إصلاح منافذ الخدمات
- ✅ تحسين واجهة المستخدم

### v1.0 - Basic Agent Service
- ✅ خدمة وكيل ذكي أساسية
- ✅ دعم قاعدة البيانات
- ✅ واجهة محادثة تفاعلية

## 📝 الترخيص | License

هذا المشروع مخصص للاستخدام التعليمي والبحثي.

This project is intended for educational and research purposes.

## 🤝 المساهمة | Contributing

نرحب بالمساهمات والتحسينات! يرجى إنشاء issue أو pull request.

We welcome contributions and improvements! Please create an issue or pull request.

---

**تم التطوير بواسطة فريق مشروع استرجاع المعلومات | Developed by the Information Retrieval Project Team** 