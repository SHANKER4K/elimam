import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar,normalize_teh_marbuta_ar,normalize_alef_maksura_ar
import torch
from functools import lru_cache
from typing import Dict, Any, List
import argparse



access_token = "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"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# Custom Embedding Function
@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=False,
            token=access_token
        )

    def __call__(self, docs: Documents) -> Embeddings:
        return self.model(docs)

    @staticmethod
    def name() -> str:
        return "GATE-AraBert-v1"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_name=self.model_name)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return MyEmbeddingFunction(config['model_name'])
    
@lru_cache(maxsize=1)
def load_assets(name):
    print('Loading Model:')
    model = MyEmbeddingFunction('Omartificial-Intelligence-Space/GATE-AraBert-v1')
    print('Model Loaded')

    # Create Client
    client = chromadb.PersistentClient(path="../Islam",settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False))

    quran_collection = client.get_or_create_collection(
        name='quran',
        embedding_function=model
    )

    tafsir_collection = client.get_or_create_collection(
        name='tafsir',
        embedding_function=model
    )

    hadith_collection = client.get_or_create_collection(
        name='hadith',
        embedding_function=model
    )

    aqeedah_collection = client.get_or_create_collection(
        name='aqeedah',
        embedding_function=model
    )
    return model,quran_collection,tafsir_collection,hadith_collection,aqeedah_collection

model,quran_collection,tafsir_collection,hadith_collection,aqeedah_collection = load_assets('Omartificial-Intelligence-Space/GATE-AraBert-v1')

def get_ayahs(surah, ayah):
    """Get verse(s) from the Quran.

    Args:
        surah: surah number (e.g. 1)
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each dict has keys: ids, documents, metadatas.
        Metadata format: {"surah": str, "surah_number": int, "ayah_number": int}
    """
    ayat = [ayah]
    if '-' in ayah:
        ayat = ayah.split('-')

    ayat = [quran_collection.get(f'{surah}:{aya}') for aya in ayat]
    return ayat

def get_hadith(book, chapter, hadith):
    """Get a specific hadith by book, chapter, and hadith number.

    Args:
        book: one of abudaud, bukhari, ibnmaja, muslim, nesai, tirmizi
        chapter: chapter number
        hadith: hadith number

    Returns:
        dict — keys: ids, documents, metadatas.
        Metadata format: {"book": str, "chapter_number": int, "chapter": str,
                         "hadith_number": int, "grade": str, "sanad": str}
    """
    return hadith_collection.get(f'{book}:{chapter}:{hadith}')

def get_tafsir(book, surah, ayah):
    """Get tafsir for a verse from one or all books.

    Args:
        book: one of tabari, ibnkathir, saadi, baghawi, chengiti, or '*' for all
        surah: surah number
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each dict has keys: ids, documents, metadatas.
        Metadata format: {"aya": str, "surah_name": str, "surah_numer": int,
                         "aya_number": int, "tafsir_book": str,
                         "tafsir_book_short": str, "era": str}
    """
    books = [book]
    if book == '*':
        books = ['tabari', 'ibnkathir', 'saadi', 'baghawi', 'chengiti']
    tafasir = [ayah]
    for book in books:
        if '-' in ayah:
            tafasir = ayah.split('-')
        tafasir = [tafsir_collection.get(f'{book}:{surah}:{aya}') for aya in tafasir]

    return tafasir

def search_quran(queries: List, k=2, where=None):
    """Semantic search over the Quran corpus.

    Args:
        queries: list of search query strings
        k: number of results to return (default 2)
        where: optional metadata filter dict

    Returns:
        dict — keys: ids, documents, metadatas, distances.
        Metadata format: {"surah": str, "surah_number": int, "ayah_number": int}
    """
    return quran_collection.query(
        query_texts=queries,
        n_results=k,
        where=where
    )
    
def search_hadith(queries: List, k=2, where=None):
    """Semantic search over the hadith corpus.

    Args:
        queries: list of search query strings
        k: number of results to return (default 2)
        where: optional metadata filter dict

    Returns:
        dict — keys: ids, documents, metadatas, distances.
        Metadata format: {"book": str, "chapter_number": int, "chapter": str,
                         "hadith_number": int, "grade": str, "sanad": str}
    """
    return hadith_collection.query(
        query_texts=queries,
        n_results=k,
        where=where
    )
    
def get_aqeedah(book_id, chunk_page=None):
    """Get a specific aqeedah chunk by book ID and optionally page.

    Args:
        book_id: book ID number
        chunk_page: page number (optional)

    Returns:
        dict — keys: ids, documents, metadatas.
        Metadata format: {"book_id": int, "book_name": str, "category_name": str,
                         "all_authors": str, "chunk_page": str, "book_pages": int,
                         "source": str}
    """
    if chunk_page:
        return aqeedah_collection.get(f'{book_id}:{chunk_page}')
    return aqeedah_collection.get(f'{book_id}:')


def search_aqeedah(queries: List, k=2, where=None):
    """Semantic search over the aqeedah corpus.

    Args:
        queries: list of search query strings
        k: number of results to return (default 2)
        where: optional metadata filter dict

    Returns:
        dict — keys: ids, documents, metadatas, distances.
        Metadata format: {"book_id": int, "book_name": str, "category_name": str,
                         "all_authors": str, "chunk_page": str, "book_pages": int,
                         "source": str}
    """
    return aqeedah_collection.query(
        query_texts=queries,
        n_results=k,
        where=where
    )
    
def main():
    parser = argparse.ArgumentParser(
        description="CLI to get or search in quran, hadith and tafsir datasets",
        epilog="""examples:
  python cli/server.py quran get 1:1
  python cli/server.py quran get 1:1-5
  python cli/server.py quran search "رحمة" -k 3
  python cli/server.py hadith get bukhari:1:1
  python cli/server.py hadith search "صلاة" -k 5
  python cli/server.py tafsir get tabari:1:1
  python cli/server.py tafsir search "نور" -k 2""")
    dataset = parser.add_subparsers(dest="command",help='Select dataset (quran,hadith,tafsir,aqeedah)')
    
    quran_parser = dataset.add_parser("quran")
    hadith_parser = dataset.add_parser("hadith")
    tafsir_parser = dataset.add_parser("tafsir")
    aqeedah_parser = dataset.add_parser("aqeedah")
    
    quran_sub = quran_parser.add_subparsers(dest="type",help='Do you want to search or get?')
    hadith_sub = hadith_parser.add_subparsers(dest="type",help='Do you want to search or get?')
    tafsir_sub = tafsir_parser.add_subparsers(dest="type",help='Do you want to search or get?')
    aqeedah_sub = aqeedah_parser.add_subparsers(dest="type",help='Do you want to search or get?')
    
    quran_get = quran_sub.add_parser("get")
    quran_search = quran_sub.add_parser("search")
    
    hadith_get = hadith_sub.add_parser("get")
    hadith_search = hadith_sub.add_parser("search")
    
    tafsir_get = tafsir_sub.add_parser("get")
    tafsir_search = tafsir_sub.add_parser("search")

    aqeedah_get = aqeedah_sub.add_parser("get")
    aqeedah_search = aqeedah_sub.add_parser("search")
    
    
    
    quran_get.add_argument('verse', type=str, help='surah:ayah (e.g. 1:1 or 1:1-5)')
    hadith_get.add_argument('hadith', type=str, help='book:chapter:hadith — books: abudaud, bukhari, ibnmaja, muslim, nesai, tirmizi')
    tafsir_get.add_argument('tafsir', type=str, help='book:surah:ayah — books: tabari, ibnkathir, saadi, baghawi, chengiti')

    aqeedah_get.add_argument('aqeedah', type=str, help='book_id[:chunk_page] (e.g. 12345 or 12345:10)')
    
    
    quran_search.add_argument('queries', type=str, help='Search query text', nargs='+')
    quran_search.add_argument('-k',default=2,type=int,help='How many retrivals do you need?')
    
    hadith_search.add_argument('queries',type=str,help='Search within the hadith dataset',nargs='+')
    hadith_search.add_argument('-k',default=2,type=int,help='How many retrivals do you need?')
    
    tafsir_search.add_argument('queries',type=str,help='Search within the tafsir dataset',nargs='+')
    tafsir_search.add_argument('-k',default=2,type=int,help='How many retrivals do you need?')

    aqeedah_search.add_argument('queries',type=str,help='Search within the aqeedah dataset',nargs='+')
    aqeedah_search.add_argument('-k',default=2,type=int,help='How many retrivals do you need?')

    args = parser.parse_args()
    
    if args.command =='quran':
        
        if args.type =='get':
            surah,verse = args.verse.split(':')
            print(get_ayahs(surah,verse))
            
        if args.type =='search':
            queries = args.queries
            k = args.k
            print(search_quran(queries,k))
            
    if args.command =='hadith':
        
        if args.type =='get':
            book,chapter,hadith = args.hadith.split(':')
            print(get_hadith(book,chapter,hadith))
            
        if args.type =='search':
            queries = args.queries
            k = args.k
            print(search_hadith(queries,k))
            
    if args.command =='tafsir':
        
        if args.type =='get':
            book,surah,verse = args.tafsir.split(':')
            print(get_tafsir(book,surah,verse))
            
        if args.type =='search':
            queries = args.queries
            k = args.k
            print(search_tafsir(queries,k))

    if args.command =='aqeedah':

        if args.type =='get':
            parts = args.aqeedah.split(':')
            book_id = parts[0]
            chunk_page = parts[1] if len(parts) > 1 else None
            print(get_aqeedah(book_id, chunk_page))

        if args.type =='search':
            queries = args.queries
            k = args.k
            print(search_aqeedah(queries,k))
    

if __name__ == "__main__":
    main()
