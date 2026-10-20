"""Fixed replies for the non-retrieval paths, in the languages we expect (falls back to English).

Deterministic text on purpose: refusals, "I don't know" and booking replies should never be improvised by the LLM.
"""
from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "medical": "I can't tell whether a treatment or surgery is safe for your personal health situation — that "
                   "needs a doctor who knows your medical history (including conditions such as diabetes). Please "
                   "discuss it with your doctor. PlacidWay can also arrange a review of your records by the clinic's "
                   "physicians: [request a free quote]({quote}).",
        "not_found": "I couldn't find that in the PlacidWay pages I have, so I won't guess. Our team can answer it for "
                     "you: [request a free quote]({quote}) or [contact PlacidWay]({contact}).",
        "out_of_scope": "Sorry, I can only help with questions about the treatments, clinics, packages and prices on "
                        "PlacidWay's pages (currently ITC Immunity Therapy Center's alternative cancer treatment in "
                        "Tijuana, Mexico, and alternative-medicine prices abroad). For anything else, please "
                        "[contact PlacidWay]({contact}).",
        "booking": "I can't book appointments or check availability. To arrange treatment, please use the "
                   "[free quote form]({quote}) or [contact PlacidWay]({contact}) — a patient coordinator will help "
                   "you. If you like, leave your name and email below and the team will reach out.",
        "quote_request": "Happy to help you get a free, personalised quote. You can use the [free quote form]({quote}), "
                         "or leave your name and email below and the PlacidWay team will contact you.",
        "greeting": "Hello! I can answer questions about ITC Immunity Therapy Center's alternative cancer treatment "
                    "packages in Tijuana, Mexico, their prices and what is included, and about alternative-medicine "
                    "prices abroad. What would you like to know?",
        "error": "Sorry, I'm having trouble answering right now. Please try again in a moment, or "
                 "[contact PlacidWay]({contact}).",
        "partial_cta": "For details that aren't listed on the page, [request a free quote]({quote}).",
        "disclaimer": "_Medical note: this information comes from PlacidWay's pages and is not medical advice. "
                      "Please discuss any treatment decision with your doctor._",
        "sources": "Sources",
    },
    "es": {
        "medical": "No puedo indicar si un tratamiento o cirugía es seguro para su situación de salud personal; eso "
                   "requiere un médico que conozca su historial (incluidas condiciones como la diabetes). Por favor, "
                   "consúltelo con su médico. PlacidWay también puede organizar que los médicos de la clínica revisen "
                   "sus informes: [solicite una cotización gratuita]({quote}).",
        "not_found": "No encontré esa información en las páginas de PlacidWay que tengo, así que no voy a adivinar. "
                     "Nuestro equipo puede ayudarle: [solicite una cotización gratuita]({quote}) o "
                     "[contacte a PlacidWay]({contact}).",
        "out_of_scope": "Lo siento, solo puedo ayudar con preguntas sobre los tratamientos, clínicas, paquetes y precios "
                        "de las páginas de PlacidWay (actualmente el tratamiento alternativo contra el cáncer de ITC "
                        "Immunity Therapy Center en Tijuana, México, y precios de medicina alternativa en el "
                        "extranjero). Para otros temas, [contacte a PlacidWay]({contact}).",
        "booking": "No puedo reservar citas ni consultar disponibilidad. Para organizar su tratamiento, use el "
                   "[formulario de cotización gratuita]({quote}) o [contacte a PlacidWay]({contact}); un coordinador "
                   "de pacientes le ayudará. Si lo desea, deje su nombre y correo abajo y el equipo le contactará.",
        "quote_request": "Con gusto le ayudo a obtener una cotización gratuita y personalizada. Use el "
                         "[formulario de cotización]({quote}) o deje su nombre y correo abajo y el equipo de PlacidWay "
                         "le contactará.",
        "greeting": "¡Hola! Puedo responder preguntas sobre los paquetes de tratamiento alternativo contra el cáncer "
                    "de ITC Immunity Therapy Center en Tijuana, México, sus precios y lo que incluyen, y sobre precios "
                    "de medicina alternativa en el extranjero. ¿Qué le gustaría saber?",
        "error": "Lo siento, tengo problemas para responder ahora. Inténtelo de nuevo en un momento o "
                 "[contacte a PlacidWay]({contact}).",
        "partial_cta": "Para detalles que no aparecen en la página, [solicite una cotización gratuita]({quote}).",
        "disclaimer": "_Nota médica: esta información proviene de las páginas de PlacidWay y no es consejo médico. "
                      "Consulte cualquier decisión de tratamiento con su médico._",
        "sources": "Fuentes",
    },
    "ar": {
        "medical": "لا يمكنني تحديد ما إذا كان علاج أو جراحة آمنًا لحالتك الصحية الشخصية، فهذا يحتاج إلى طبيب يعرف "
                   "تاريخك الطبي (بما في ذلك حالات مثل السكري). يرجى مناقشة ذلك مع طبيبك. يمكن لـ PlacidWay أيضًا "
                   "ترتيب مراجعة تقاريرك من قبل أطباء العيادة: [اطلب عرض سعر مجاني]({quote}).",
        "not_found": "لم أجد هذه المعلومة في صفحات PlacidWay المتاحة لدي، لذلك لن أخمّن. يمكن لفريقنا مساعدتك: "
                     "[اطلب عرض سعر مجاني]({quote}) أو [تواصل مع PlacidWay]({contact}).",
        "out_of_scope": "عذرًا، يمكنني فقط الإجابة عن الأسئلة المتعلقة بالعلاجات والعيادات والباقات والأسعار الموجودة في "
                        "صفحات PlacidWay (حاليًا العلاج البديل للسرطان في مركز ITC Immunity Therapy Center في تيخوانا، "
                        "المكسيك، وأسعار الطب البديل في الخارج). لأي موضوع آخر، [تواصل مع PlacidWay]({contact}).",
        "booking": "لا يمكنني حجز المواعيد أو التحقق من التوفر. لترتيب العلاج، استخدم [نموذج عرض السعر المجاني]({quote}) "
                   "أو [تواصل مع PlacidWay]({contact}) وسيساعدك منسق المرضى. إن رغبت، اترك اسمك وبريدك الإلكتروني "
                   "أدناه وسيتواصل معك الفريق.",
        "quote_request": "يسعدني مساعدتك في الحصول على عرض سعر مجاني ومخصص. استخدم [نموذج عرض السعر]({quote}) أو اترك "
                         "اسمك وبريدك الإلكتروني أدناه وسيتواصل معك فريق PlacidWay.",
        "greeting": "مرحبًا! يمكنني الإجابة عن أسئلتك حول باقات العلاج البديل للسرطان في مركز ITC Immunity Therapy "
                    "Center في تيخوانا، المكسيك، وأسعارها وما تشمله، وكذلك أسعار الطب البديل في الخارج. ماذا تود أن تعرف؟",
        "error": "عذرًا، أواجه مشكلة في الإجابة الآن. حاول مرة أخرى بعد قليل أو [تواصل مع PlacidWay]({contact}).",
        "partial_cta": "للتفاصيل غير المذكورة في الصفحة، [اطلب عرض سعر مجاني]({quote}).",
        "disclaimer": "_ملاحظة طبية: هذه المعلومات مأخوذة من صفحات PlacidWay وليست نصيحة طبية. "
                      "يرجى مناقشة أي قرار علاجي مع طبيبك._",
        "sources": "المصادر",
    },
    "ur": {
        "medical": "میں یہ نہیں بتا سکتا کہ کوئی علاج یا سرجری آپ کی ذاتی صحت کے لیے محفوظ ہے یا نہیں — اس کے لیے "
                   "ایسا ڈاکٹر چاہیے جو آپ کی میڈیکل ہسٹری (بشمول ذیابیطس جیسی بیماریوں) سے واقف ہو۔ براہ کرم اپنے ڈاکٹر "
                   "سے مشورہ کریں۔ PlacidWay کلینک کے ڈاکٹروں سے آپ کی رپورٹس کا جائزہ بھی کروا سکتا ہے: "
                   "[مفت کوٹ حاصل کریں]({quote})۔",
        "not_found": "مجھے یہ معلومات PlacidWay کے دستیاب صفحات میں نہیں ملی، اس لیے میں اندازہ نہیں لگاؤں گا۔ ہماری "
                     "ٹیم آپ کی مدد کر سکتی ہے: [مفت کوٹ حاصل کریں]({quote}) یا [PlacidWay سے رابطہ کریں]({contact})۔",
        "out_of_scope": "معذرت، میں صرف PlacidWay کے صفحات پر موجود علاج، کلینکس، پیکجز اور قیمتوں کے بارے میں سوالات میں "
                        "مدد کر سکتا ہوں (فی الحال تیخوانا، میکسیکو میں ITC Immunity Therapy Center کا متبادل کینسر علاج، "
                        "اور بیرون ملک متبادل طب کی قیمتیں)۔ کسی اور معاملے کے لیے [PlacidWay سے رابطہ کریں]({contact})۔",
        "booking": "میں اپوائنٹمنٹ بک نہیں کر سکتا اور نہ ہی دستیابی چیک کر سکتا ہوں۔ علاج کے انتظام کے لیے "
                   "[مفت کوٹ فارم]({quote}) استعمال کریں یا [PlacidWay سے رابطہ کریں]({contact})۔ اگر آپ چاہیں تو نیچے "
                   "اپنا نام اور ای میل دیں، ٹیم آپ سے رابطہ کرے گی۔",
        "quote_request": "مفت اور ذاتی کوٹ حاصل کرنے میں خوشی سے مدد کروں گا۔ [کوٹ فارم]({quote}) استعمال کریں یا نیچے "
                         "اپنا نام اور ای میل دیں، PlacidWay کی ٹیم آپ سے رابطہ کرے گی۔",
        "greeting": "السلام علیکم! میں تیخوانا، میکسیکو میں ITC Immunity Therapy Center کے متبادل کینسر علاج کے پیکجز، "
                    "ان کی قیمتوں اور ان میں شامل سہولیات، اور بیرون ملک متبادل طب کی قیمتوں کے بارے میں سوالات کے "
                    "جواب دے سکتا ہوں۔ آپ کیا جاننا چاہیں گے؟",
        "error": "معذرت، اس وقت جواب دینے میں مسئلہ ہو رہا ہے۔ تھوڑی دیر بعد دوبارہ کوشش کریں یا "
                 "[PlacidWay سے رابطہ کریں]({contact})۔",
        "partial_cta": "صفحے پر درج نہ ہونے والی تفصیلات کے لیے [مفت کوٹ حاصل کریں]({quote})۔",
        "disclaimer": "_طبی نوٹ: یہ معلومات PlacidWay کے صفحات سے لی گئی ہیں اور طبی مشورہ نہیں ہیں۔ علاج کا کوئی بھی "
                      "فیصلہ اپنے ڈاکٹر سے مشورہ کر کے کریں۔_",
        "sources": "ذرائع",
    },
}


def text(key: str, language: str, **links: str) -> str:
    table = TEMPLATES.get(language, TEMPLATES["en"])
    return table.get(key, TEMPLATES["en"][key]).format(**links)
