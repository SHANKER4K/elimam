# ER Diagram — PostgreSQL

Source of truth: `frontend/src/db/schema.ts` (Drizzle). The backend has no DDL and no migrations; it queries these tables with raw psycopg2 SQL.

```mermaid
erDiagram
  users ||--o{ session : "better-auth session"
  users ||--o{ account : "oauth/password"
  users ||--o{ sessions : "agent conversations"
  users ||--o{ user_providers : "connections"
  users ||--o{ api_keys : "frozen legacy snapshot"
  sessions ||--o{ messages : "projected turns"
  user_providers ||--o{ provider_models : "discovered/added models"
  user_providers }o--|| providers : "provider_id (NULL = custom)"
  sessions }o--o| user_providers : "user_provider_id (SET NULL)"
  providers ||--o{ user_providers : "catalog entry"

  users {
    uuid id PK
    text telegram_id UK "nullable; bot identity"
    text username
    text display_name
    text image
    varchar email UK
    boolean email_verified
    text role "better-auth admin plugin"
    boolean banned
    timestamp ban_expires
    timestamptz created_at
    timestamptz updated_at
  }

  session {
    text id PK "better-auth (text, not uuid)"
    uuid user_id FK
    varchar token UK
    timestamptz expires_at
    text ip_address
    text user_agent
    text impersonated_by
    timestamptz created_at
    timestamptz updated_at
  }

  account {
    text id PK
    uuid user_id FK
    text issuer
    text account_id
    text provider_id
    text access_token
    text refresh_token
    text scope
    text id_token
    text password
  }

  verification {
    text id PK
    text identifier
    text value
    timestamptz expires_at
  }

  sessions {
    uuid id PK
    uuid user_id FK
    text source "telegram or web"
    text model_provider
    text model_name
    text model_variant
    uuid user_provider_id FK "nullable, on delete set null"
    jsonb pydantic_message "complete pydantic-ai message history"
    boolean is_active "partial unique index: one active per user"
    timestamptz created_at
    timestamptz updated_at
  }

  messages {
    uuid id PK
    uuid session_id FK
    message_role role "enum: user/assistant/system/tool"
    text content
    integer sequence "computed via MAX(sequence)+offset in INSERT"
    timestamptz created_at
  }

  api_keys {
    uuid id PK
    uuid user_id FK
    text provider
    text encrypted_key
  }

  providers {
    serial id PK
    text slug UK "openai, anthropic, groq, omniroute..."
    text name
    provider_api_style api_style "openai or anthropic compatible"
    text default_base_url
    text logo_url
    text_array default_variants "defaults to low"
    boolean requires_key "false for local keyless gateways"
    text docs_url
    jsonb models "catalog truth, model id to variants map"
    timestamptz created_at
  }

  user_providers {
    uuid id PK
    uuid user_id FK
    integer provider_id FK "NULL = fully custom endpoint"
    boolean is_custom
    text custom_name
    text base_url "required"
    provider_api_style api_style
    text encrypted_key "NULL only when requires_key=false"
    jsonb extra_headers
    timestamptz last_validated_at
    timestamptz created_at
    timestamptz updated_at
  }

  provider_models {
    uuid id PK
    uuid user_provider_id FK
    text model_id
    text display_name
    text_array variants
    integer context_window
    boolean supports_tools
    boolean supports_vision
    boolean is_custom
    boolean enabled "soft delete: hidden from pickers, still resolvable"
    timestamptz last_synced_at
  }

  surahs ||--o{ ayahs : "surah_id"
  hadith_books ||--o{ hadith_chapters : "book_id"
  hadith_books ||--o{ hadiths : "book_id"
  hadith_chapters ||--o{ hadiths : "chapter_id"
  themes ||--o{ theme_ayahs : "theme_id"
  themes ||--o{ theme_hadiths : "theme_id"
  ayahs ||--o{ theme_ayahs : "ayah_id"
  hadiths ||--o{ theme_hadiths : "hadith_id"

  surahs {
    serial id PK
    integer number UK
    text name_ar
    text name_translation
    integer verses_count
    revelation_type revelation_type "Meccan or Medinan"
  }

  ayahs {
    serial id PK
    integer surah_id FK
    integer number_in_surah
    text text_uthmani
    text text_simple
    text text_en
    text tafsir_text
    text asbab_nuzul
  }

  hadith_books {
    serial id PK
    text name_ar
    text name_en
    text slug UK
  }

  hadith_chapters {
    serial id PK
    integer book_id FK
    text name_ar
    text name_en
    integer order
  }

  hadiths {
    serial id PK
    integer chapter_id FK
    integer book_id FK
    integer number
    text narrator
    text text
    text text_simple
    text text_en
    hadith_grade grade "Sahih, Hasan or Dhaeef"
    text sharh
  }

  hadiths_with_sanad_matn {
    integer id
    integer chapter_id
    integer book_id
    integer number
    text text
    text sanad
    text matn
  }

  themes {
    serial id PK
    text slug UK
    text name_ar
    text name_en
    text description
    theme_status status "draft or published"
  }

  theme_ayahs {
    serial id PK
    integer theme_id FK
    integer ayah_id FK
    text reviewed_by
    text note
  }

  theme_hadiths {
    serial id PK
    integer theme_id FK
    integer hadith_id FK
    text reviewed_by
    text note
  }
```

Two "session" tables, and they are unrelated:

- `session` / `account` / `verification` / `verification` = Better Auth (web login). `session.id` is **text**, ids come from `crypto.randomUUID()`.
- `sessions` = agent conversations the backend owns (`user_id`, model triple, `pydantic_message`). `messages.session_id` points here, not at `session`.

Other constraints worth remembering:

- `sessions_one_active_per_user_idx` is a partial unique index on `(user_id) WHERE is_active = true`. Web sessions insert with `is_active = false` to stay out of it.
- `user_providers` has a unique index on `(user_id, provider_id)`; NULLs never collide, so a user can hold unlimited custom connections but one per catalog provider.
- `api_keys` is frozen (rollback snapshot for the `user_providers` migration) — never read or written by current code.
- `hadiths_with_sanad_matn` is created outside Drizzle and read-only.
