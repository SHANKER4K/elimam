# %%
from camel_tools.utils.dediac import dediac_ar
from IPython.display import display,Markdown
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# %%

embed_model = HuggingFaceEmbedding(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")

splitter = SemanticSplitterNodeParser(
    buffer_size=2, # Contextual buffer (number of sentences) around split points
    breakpoint_percentile_threshold=75, # Sensitivity to semantic shifts
    embed_model=embed_model
)
# %%
from llama_index.core import Document
documents = [Document(id=1,text="""باب المقدمات
مقدمة التحقيق
... ﷽ مقدمة التحقيق الحمد الله القائل: ﴿وَأَنَّ هَذَا صِرَاطِي مُسْتَقِيمًا فَاتَّبِعُوهُ وَلا تَتَّبِعُوا السُّبُلَ فَتَفَرَّقَ بِكُمْ عَنْ سَبِيلِهِ ذَلِكُمْ وَصَّاكُمْ بِهِ لَعَلَّكُمْ تَتَّقُونَ﴾ [الأنعام: ١٥٣] والقائل سبحانه: ﴿وَلا تَكُونُوا مِنَ الْمُشْرِكِينَ، مِنَ الَّذِينَ فَرَّقُوا دِينَهُمْ وَكَانُوا شِيَعًا كُلُّ حِزْبٍ بِمَا لَدَيْهِمْ فَرِحُون﴾ [الروم: ٣١] القائل ﷿: ﴿وَلا تَكُونُوا كَالَّذِينَ تَفَرَّقُوا وَاخْتَلَفُوا مِنْ بَعْدِ مَا جَاءَهُمُ الْبَيِّنَاتُ وَأُوْلَئِكَ لَهُمْ عَذَابٌ عَظِيمٌ﴾ [آل عمران: ١٠٥] وأصلي وأسلم على النبي الرحمة المهداة محمد بن عبد الله القائل: "فإن خير الحديث كتاب الله، وخير الهدي هدي محمد، وشر الأمور محدثاتها، وكل بعدة ضلالة" ١ والقائل ﷺ: "فإنه من يعش منكم بعدي سيرى اختلافًا كثيرًا، فعليكم بسنتي وسنة الخلفاء الراشدين المهديين، عَضوا عليها بالنواجذ، وإياكم ومحدثات الأمور، فإن كل محدثة بدعة، وكل بدعة ضلالة" ٢. ورحم الله الأوزاعي حين قال: عليك بأثر من سلف وإن رفضك الناس، وإياك وآراء الرجال وإن زخرفوا لك بالقول.

nodes = splitter.get_nodes_from_documents(documents)
""")]




splitter = SemanticSplitterNodeParser(
    buffer_size=2, # Contextual buffer (number of sentences) around split points
    breakpoint_percentile_threshold=75, # Sensitivity to semantic shifts
    embed_model=embed_model
)

# %%
nodes = splitter.get_nodes_from_documents(documents)

# %%
for node in nodes:
    display(Markdown(node.text))