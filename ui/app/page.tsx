"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Search,
  BookOpen,
  ExternalLink,
  SlidersHorizontal,
  Blend,
} from "lucide-react";
import type {
  ChromaResult,
  SearchResult,
  HybridResultItem,
  HadithBook,
  TafsirBook,
} from "@/lib/api";
import * as api from "@/lib/api";

// ── helpers ──

function fmtMeta(meta: Record<string, unknown> | undefined) {
  if (!meta) return [];
  return Object.entries(meta).filter(([_, v]) => v != null);
}

function KeyVal({ k, v }: { k: string; v: unknown }) {
  return (
    <Badge variant="outline" className="gap-1 text-xs">
      <span className="text-muted-foreground">{k}:</span>
      <span>{String(v)}</span>
    </Badge>
  );
}

// ── search result card ──

function SearchResultCard({
  doc,
  meta,
  dist,
}: {
  doc: string;
  meta?: Record<string, unknown>;
  dist?: number;
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-1.5">
          {fmtMeta(meta)
            .slice(0, 4)
            .map(([k, v]) => (
              <KeyVal key={k} k={k} v={v} />
            ))}
          {dist != null && (
            <Badge variant="secondary">distance: {dist.toFixed(4)}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p dir="auto" className="leading-relaxed text-foreground/90">
          {doc}
        </p>
      </CardContent>
    </Card>
  );
}

// ── get result card ──

function GetResultCard({ result }: { result: ChromaResult }) {
  if (!result.documents?.length) return null;
  return (
    <div className="flex flex-col gap-3">
      {result.documents.map((doc, i) => (
        <Card key={i} size="sm">
          {result.metadatas?.[i] && (
            <CardHeader>
              <div className="flex flex-wrap gap-1.5">
                {fmtMeta(result.metadatas[i]).map(([k, v]) => (
                  <KeyVal key={k} k={k} v={v} />
                ))}
              </div>
            </CardHeader>
          )}
          <CardContent>
            <p dir="auto" className="leading-relaxed">
              {doc}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ── hybrid result card ──

function HybridResultCard({ item }: { item: HybridResultItem }) {
  const badges: { k: string; v: string }[] = [];
  if (item.surah) badges.push({ k: "surah", v: item.surah });
  if (item.ayah_number != null)
    badges.push({ k: "ayah", v: String(item.ayah_number) });
  if (item.surah_number != null)
    badges.push({ k: "surah#", v: String(item.surah_number) });
  if (item.book) badges.push({ k: "book", v: item.book });
  if (item.book_full) badges.push({ k: "Book Name", v: item.book_full });
  if (item.grade) badges.push({ k: "grade", v: item.grade });
  if (item.hadith_number) badges.push({ k: "hadith#", v: item.hadith_number });
  if (item.tafsir_book) badges.push({ k: "tafsir", v: item.tafsir_book });
  if (item.source) badges.push({ k: "source", v: item.source });
  if (item.section) badges.push({ k: "section", v: item.section });
  if (item.section_number)
    badges.push({ k: "sectin_number", v: item.section_number });
  // aqeedah fields
  if (item.book_name) badges.push({ k: "book", v: item.book_name });
  if (item.category_name)
    badges.push({ k: "category", v: item.category_name });
  if (item.all_authors) badges.push({ k: "author", v: item.all_authors });
  if (item.chunk_page) badges.push({ k: "page", v: item.chunk_page });
  if (item.book_pages != null)
    badges.push({ k: "total_pages", v: String(item.book_pages) });
  if (item.book_id != null)
    badges.push({ k: "book_id", v: String(item.book_id) });

  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-1.5">
          {badges.map(({ k, v }) => (
            <Badge key={k} variant="outline" className="gap-1 text-xs">
              <span className="text-muted-foreground">{k}:</span>
              <span>{v}</span>
            </Badge>
          ))}
          {/* scores */}
          <div className="flex gap-1 text-xs text-muted-foreground ml-auto">
            <span title="Hybrid score">
              H:{(item.hybrid_score * 100).toFixed(0)}
            </span>
            <span title="Semantic score">
              · S:{(item.semantic_score * 100).toFixed(0)}
            </span>
            <span title="Keyword score">
              · K:{(item.keyword_score * 100).toFixed(0)}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p dir="auto" className="leading-relaxed text-foreground/90">
          {item.text}
        </p>
      </CardContent>
    </Card>
  );
}

// ══════════════════════  SEARCH FORM  ══════════════════════

function SearchPanel() {
  const [searchMode, setSearchMode] = useState<"semantic" | "hybrid">(
    "semantic",
  );
  const [corpus, setCorpus] = useState("quran");
  const [query, setQuery] = useState("");
  const [k, setK] = useState("5");
  const [alpha, setAlpha] = useState("0.6");
  const [where, setWhere] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [hybridResult, setHybridResult] = useState<HybridResultItem[] | null>(
    null,
  );

  const whereHint: Record<string, string> = {
    quran: '{"surah_number": 1}',
    hadith: '{"book": "bukhary"}',
    tafsir: '{"tafsir_book": "tabary"}',
  };

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    setHybridResult(null);
    try {
      if (searchMode === "hybrid") {
        const w = where.trim() || undefined;
        const res =
          corpus === "aqeedah"
            ? await api.hybridSearchAqeedah(query, Number(k), Number(alpha))
            : corpus === "quran"
              ? await api.hybridSearchQuran(query, Number(k), Number(alpha))
              : corpus === "hadith"
                ? await api.hybridSearchHadith(query, Number(k), Number(alpha))
                : await api.hybridSearchTafsir(query, Number(k), Number(alpha));
        setHybridResult(res);
      } else {
        const w = where.trim() || undefined;
        const res =
          corpus === "aqeedah"
            ? await api.searchAqeedah(query, Number(k), w)
            : corpus === "quran"
              ? await api.searchQuran(query, Number(k), w)
              : corpus === "hadith"
                ? await api.searchHadith(query, Number(k), w)
                : await api.searchTafsir(query, Number(k), w);
        setResult(res);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  const hasResults =
    searchMode === "hybrid"
      ? hybridResult && hybridResult.length > 0
      : result && result.documents?.[0]?.length;

  return (
    <div className="flex flex-col gap-4">
      <Tabs
        value={corpus}
        onValueChange={(v) => {
          if (v) setCorpus(v);
          setResult(null);
          setError("");
        }}
      >
        <TabsList className="w-full">
          <TabsTrigger value="quran" className="flex-1">
            Quran
          </TabsTrigger>
          <TabsTrigger value="hadith" className="flex-1">
            Hadith
          </TabsTrigger>
          <TabsTrigger value="tafsir" className="flex-1">
            Tafsir
          </TabsTrigger>
          <TabsTrigger value="aqeedah" className="flex-1">
            Aqeedah
          </TabsTrigger>
        </TabsList>

        {/* inputs outside TabsContent so they don't swap on every switch */}
      </Tabs>

      {/* ── mode toggle: Semantic vs Hybrid ── */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => {
            setSearchMode("semantic");
            setResult(null);
            setHybridResult(null);
          }}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
            searchMode === "semantic"
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          }`}
        >
          Semantic
        </button>
        <button
          onClick={() => {
            setSearchMode("hybrid");
            setResult(null);
            setHybridResult(null);
          }}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
            searchMode === "hybrid"
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          }`}
        >
          <Blend className="size-3.5 inline mr-1 align-text-bottom" />
          Hybrid
        </button>
      </div>

      <div className="flex flex-col gap-3">
        <Input
          placeholder={
            corpus === "quran"
              ? "e.g. التوكل على الله"
              : corpus === "hadith"
                ? "e.g. فضل الصلاة"
                : "e.g. تفسير الرحمن"
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          dir="auto"
        />

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Results:</label>
            <Input
              type="number"
              min={1}
              max={50}
              className="w-16 h-7 text-xs"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </div>

          {searchMode === "hybrid" && (
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="size-3.5 text-muted-foreground" />
              <label className="text-xs text-muted-foreground">
                Alpha: {alpha}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={alpha}
                onChange={(e) => setAlpha(e.target.value)}
                className="w-20 h-1.5 accent-primary cursor-pointer"
              />
              <span className="text-[10px] text-muted-foreground/60">
                K:0 · S:1
              </span>
            </div>
          )}

          <Button onClick={handleSearch} disabled={loading || !query.trim()}>
            <Search data-icon="inline-start" />
            {loading ? "Searching…" : "Search"}
          </Button>
        </div>

        <details className="group text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            Advanced filters (where JSON)
          </summary>
          <div className="mt-2">
            <Textarea
              placeholder={whereHint[corpus]}
              value={where}
              onChange={(e) => setWhere(e.target.value)}
              className="font-mono text-xs min-h-12"
              rows={2}
            />
          </div>
        </details>
      </div>

      <Separator />

      {/* results */}
      {error && <p className="text-destructive text-sm">{error}</p>}

      {loading && (
        <div className="flex flex-col gap-3 overflow-y-auto">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} size="sm">
              <CardHeader>
                <Skeleton className="h-4 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-12 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!loading && searchMode === "semantic" && hasResults && (
        <ScrollArea className="max-h-[60vh] overflow-y-auto">
          <div className="flex flex-col gap-3 pr-3">
            {result!.documents[0].map((doc, i) => (
              <SearchResultCard
                key={result!.ids[0][i]}
                doc={doc}
                meta={result!.metadatas[0][i]}
                dist={result!.distances?.[0]?.[i]}
              />
            ))}
          </div>
        </ScrollArea>
      )}

      {!loading &&
        searchMode === "hybrid" &&
        hybridResult &&
        hybridResult.length > 0 && (
          <ScrollArea className="max-h-[60vh] overflow-y-auto">
            <div className="flex flex-col gap-3 pr-3">
              {hybridResult.map((item) => (
                <HybridResultCard key={item.id} item={item} />
              ))}
            </div>
          </ScrollArea>
        )}

      {!loading && result && !hasResults && (
        <p className="text-sm text-muted-foreground">No results found.</p>
      )}
    </div>
  );
}

// ══════════════════════  GET FORM  ══════════════════════

function GetPanel() {
  const [mode, setMode] = useState("ayah");

  // ayah
  const [surah, setSurah] = useState("");
  const [ayah, setAyah] = useState("");

  // hadith
  const [hadithBook, setHadithBook] = useState<HadithBook>("bukhari");
  const [chapter, setChapter] = useState("");
  const [hadithNum, setHadithNum] = useState("");

  // tafsir
  const [tafsirBook, setTafsirBook] = useState<TafsirBook | "*">("*");
  const [tafsirSurah, setTafsirSurah] = useState("");
  const [tafsirAyah, setTafsirAyah] = useState("");

  // aqeedah
  const [aqeedahBookId, setAqeedahBookId] = useState("");
  const [aqeedahChunkPage, setAqeedahChunkPage] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<
    ChromaResult[] | ChromaResult | api.AqeedahResult | null
  >(null);

  async function handleGet() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      if (mode === "ayah") {
        setResult(await api.getAyah(surah, ayah));
      } else if (mode === "hadith") {
        setResult(await api.getHadith(hadithBook, hadithNum));
      } else if (mode === "aqeedah") {
        setResult(
          await api.getAqeedah(aqeedahBookId, aqeedahChunkPage || undefined),
        );
      } else {
        setResult(await api.getTafsir(tafsirBook, tafsirSurah, tafsirAyah));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Tabs
        value={mode}
        onValueChange={(v) => {
          if (v) setMode(v);
          setResult(null);
          setError("");
        }}
      >
        <TabsList className="w-full">
          <TabsTrigger value="ayah" className="flex-1">
            Ayah
          </TabsTrigger>
          <TabsTrigger value="hadith" className="flex-1">
            Hadith
          </TabsTrigger>
          <TabsTrigger value="tafsir" className="flex-1">
            Tafsir
          </TabsTrigger>
          <TabsTrigger value="aqeedah" className="flex-1">
            Aqeedah
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="flex flex-col gap-3">
        {/* ── Ayah form ── */}
        {mode === "ayah" && (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-muted-foreground">Surah</label>
                <Input
                  placeholder="e.g. 1"
                  value={surah}
                  onChange={(e) => setSurah(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleGet()}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-muted-foreground">
                  Ayah (or range)
                </label>
                <Input
                  placeholder="e.g. 1 or 1-5"
                  value={ayah}
                  onChange={(e) => setAyah(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleGet()}
                />
              </div>
            </div>
            <div>
              <Button onClick={handleGet} disabled={loading || !surah || !ayah}>
                <BookOpen data-icon="inline-start" />
                {loading ? "…" : "Get"}
              </Button>
            </div>
          </>
        )}

        {/* ── Hadith form ── */}
        {mode === "hadith" && (
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Book</label>
              <Select
                value={hadithBook}
                onValueChange={(v) => v && setHadithBook(v as HadithBook)}
              >
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(
                    [
                      "abudawud",
                      "bukhari",
                      "dehlawi",
                      "ibnmajah",
                      "malik",
                      "nasai",
                      "nawawi",
                      "qudsi",
                      "tirmidhi",
                    ] as const
                  ).map((b) => (
                    <SelectItem key={b} value={b}>
                      {b}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {/* <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Chapter</label>
              <Input
                value={chapter}
                onChange={(e) => setChapter(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-20"
              />
            </div> */}
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Hadith #</label>
              <Input
                value={hadithNum}
                onChange={(e) => setHadithNum(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-20"
              />
            </div>
            <Button onClick={handleGet} disabled={loading || !hadithNum}>
              <BookOpen data-icon="inline-start" />
              {loading ? "…" : "Get"}
            </Button>
          </div>
        )}

        {/* ── Aqeedah form ── */}
        {mode === "aqeedah" && (
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Book ID</label>
              <Input
                placeholder="e.g. 12345"
                value={aqeedahBookId}
                onChange={(e) => setAqeedahBookId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-24"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Page (optional)
              </label>
              <Input
                placeholder="e.g. 10"
                value={aqeedahChunkPage}
                onChange={(e) => setAqeedahChunkPage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-20"
              />
            </div>
            <Button onClick={handleGet} disabled={loading || !aqeedahBookId}>
              <BookOpen data-icon="inline-start" />
              {loading ? "…" : "Get"}
            </Button>
          </div>
        )}

        {/* ── Tafsir form ── */}
        {mode === "tafsir" && (
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Book</label>
              <Select
                value={tafsirBook}
                onValueChange={(v) => v && setTafsirBook(v as TafsirBook | "*")}
              >
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="*">All (*)</SelectItem>
                  {(
                    [
                      "tabary",
                      "katheer",
                      "moyassar",
                      "saadi",
                      "baghawy",
                    ] as const
                  ).map((b) => (
                    <SelectItem key={b} value={b}>
                      {b}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Surah</label>
              <Input
                placeholder="e.g. 1"
                value={tafsirSurah}
                onChange={(e) => setTafsirSurah(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-20"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Ayah (or range)
              </label>
              <Input
                placeholder="e.g. 1-3"
                value={tafsirAyah}
                onChange={(e) => setTafsirAyah(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGet()}
                className="w-24"
              />
            </div>
            <Button
              onClick={handleGet}
              disabled={loading || !tafsirSurah || !tafsirAyah}
            >
              <BookOpen data-icon="inline-start" />
              {loading ? "…" : "Get"}
            </Button>
          </div>
        )}
      </div>

      <Separator />

      {error && <p className="text-destructive text-sm">{error}</p>}

      {loading && (
        <div className="flex flex-col gap-3 overflow-y-auto">
          {Array.from({ length: 2 }).map((_, i) => (
            <Card key={i} size="sm">
              <CardHeader>
                <Skeleton className="h-4 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-12 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!loading && result && (
        <ScrollArea className="max-h-[60vh] overflow-y-auto">
          <div className="flex flex-col gap-3 pr-3">
            {Array.isArray(result) ? (
              result.map((r, i) => <GetResultCard key={i} result={r} />)
            ) : (
              <GetResultCard result={result} />
            )}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}

// ══════════════════════  PAGE  ══════════════════════

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-4 pt-8 sm:p-8">
      {/* header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-heading font-semibold tracking-tight">
          فهرس التراث · Turath Index
        </h1>
        <p className="text-sm text-muted-foreground">
          Quran · Hadith · Tafsir — semantic search & retrieval
        </p>
      </div>

      {/* main tabs */}
      <Tabs defaultValue="search">
        <TabsList className="w-full">
          <TabsTrigger value="search" className="flex-1 gap-2">
            <Search className="size-4" data-icon="inline-start" />
            Search
          </TabsTrigger>
          <TabsTrigger value="get" className="flex-1 gap-2">
            <ExternalLink className="size-4" data-icon="inline-start" />
            Get
          </TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Search</CardTitle>
            </CardHeader>
            <CardContent>
              <SearchPanel />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="get" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Exact Retrieval</CardTitle>
            </CardHeader>
            <CardContent>
              <GetPanel />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
