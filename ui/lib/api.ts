const BASE = process.env.NEXT_PUBLIC_API_URL ?? "/turath";

async function get<T>(
  path: string,
  params: Record<string, string>,
): Promise<T> {
  const url = new URL(path, BASE);
  for (const [k, v] of Object.entries(params)) {
    if (v) url.searchParams.set(k, v);
  }
  const res = await fetch(url);
  console.log(`GET ${url} => ${res.status}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

// ---- types ----

export interface ChromaResult {
  ids: string[];
  documents: string[];
  metadatas: Record<string, unknown>[];
}

export interface SearchResult {
  ids: string[][];
  distances: number[][];
  documents: string[][];
  metadatas: Record<string, unknown>[][];
}

export interface AqeedahResult {
  ids: string[];
  documents: string[];
  metadatas: Record<string, unknown>[];
}

export type HadithBook =
  | "abudaud"
  | "bukhari"
  | "ibnmaja"
  | "muslim"
  | "nesai"
  | "tirmizi";
export type TafsirBook =
  | "tabari"
  | "ibnkathir"
  | "saadi"
  | "baghawi"
  | "chengiti";

// ---- get endpoints ----

export function getAqeedah(book_id: string, chunk_page?: string) {
  const params: Record<string, string> = { book_id };
  if (chunk_page) params.chunk_page = chunk_page;
  return get<AqeedahResult>("get-aqeedah", params);
}

export function getAyah(surah: string, ayah: string) {
  return get<ChromaResult[]>("get-ayah", { surah, ayah });
}

export function getHadith(book: HadithBook, hadith_number: string) {
  return get<ChromaResult>("get-hadith", { book, hadith_number });
}

export function getTafsir(book: TafsirBook | "*", surah: string, ayah: string) {
  return get<ChromaResult[]>("get-tafsir", { book, surah, ayah });
}

// ---- search endpoints ----

export function searchQuran(
  query: string,
  k = 5,
  where?: string,
  whereDocument?: string,
) {
  return get<SearchResult>("quran-search", {
    query,
    k: String(k),
    ...(where ? { where } : {}),
    ...(whereDocument ? { where_document: whereDocument } : {}),
  });
}

export function searchHadith(
  query: string,
  k = 5,
  where?: string,
  whereDocument?: string,
) {
  return get<SearchResult>("hadith-search", {
    query,
    k: String(k),
    ...(where ? { where } : {}),
    ...(whereDocument ? { where_document: whereDocument } : {}),
  });
}

export function searchAqeedah(
  query: string,
  k = 5,
  where?: string,
  whereDocument?: string,
) {
  return get<SearchResult>("aqeedah-search", {
    query,
    k: String(k),
    ...(where ? { where } : {}),
    ...(whereDocument ? { where_document: whereDocument } : {}),
  });
}

export function searchTafsir(
  query: string,
  k = 5,
  where?: string,
  whereDocument?: string,
) {
  return get<SearchResult>("tafsir-search", {
    query,
    k: String(k),
    ...(where ? { where } : {}),
    ...(whereDocument ? { where_document: whereDocument } : {}),
  });
}

// ---- hybrid search types & endpoints ----

export interface HybridResultItem {
  id: string;
  text: string;
  hybrid_score: number;
  keyword_score: number;
  semantic_score: number;
  surah?: string;
  surah_number?: number;
  ayah_number?: number;
  book?: string;
  book_full?: string;
  grade?: string;
  hadith_number?: string;
  tafsir_book?: string;
  source?: string;
  section?: string;
  section_number?: string;
  // aqeedah fields
  book_id?: number;
  book_name?: string;
  category_name?: string;
  all_authors?: string;
  chunk_page?: string;
  book_pages?: number;
}

export type HybridSearchResult = HybridResultItem[];

export function hybridSearchQuran(query: string, k = 10, alpha = 0.6) {
  return get<HybridSearchResult>("hybrid-quran-search", {
    query,
    k: String(k),
    alpha: String(alpha),
  });
}

export function hybridSearchHadith(query: string, k = 10, alpha = 0.6) {
  return get<HybridSearchResult>("hybrid-hadith-search", {
    query,
    k: String(k),
    alpha: String(alpha),
  });
}

export function hybridSearchAqeedah(query: string, k = 10, alpha = 0.6) {
  return get<HybridSearchResult>("hybrid-aqeedah-search", {
    query,
    k: String(k),
    alpha: String(alpha),
  });
}

export function hybridSearchTafsir(query: string, k = 10, alpha = 0.6) {
  return get<HybridSearchResult>("hybrid-tafsir-search", {
    query,
    k: String(k),
    alpha: String(alpha),
  });
}
