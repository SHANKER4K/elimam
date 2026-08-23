import { sql } from "drizzle-orm";
import {
  pgTable,
  uuid,
  text,
  timestamp,
  jsonb,
  integer,
  boolean,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";

// ====================
// Users
// ====================

export const users = pgTable(
  "users",
  {
    id: uuid("id").defaultRandom().primaryKey(),

    // Telegram-specific identifier. Nullable so other clients (web, etc.)
    // can create users without a Telegram identity.
    telegramId: text("telegram_id"),

    username: text("username"),
    displayName: text("display_name"),
    email: text("email"),

    createdAt: timestamp("created_at", { withTimezone: true })
      .defaultNow()
      .notNull(),

    updatedAt: timestamp("updated_at", { withTimezone: true })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    uniqueIndex("users_telegram_id_idx").on(table.telegramId),
    uniqueIndex("users_email_idx").on(table.email),
  ],
);

// ====================
// API Keys
// ====================
// Belongs to the USER, not the session. One key per (user, provider).

export const apiKeys = pgTable(
  "api_keys",
  {
    id: uuid("id").defaultRandom().primaryKey(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    provider: text("provider").notNull(),

    // Isolated behind this single column so real encryption (e.g. AES-GCM
    // with a KMS-managed key) can be dropped in later without touching
    // callers. Until then this holds the raw key value - never log it,
    // never return it in API responses.
    encryptedKey: text("encrypted_key").notNull(),

    createdAt: timestamp("created_at", { withTimezone: true })
      .defaultNow()
      .notNull(),

    updatedAt: timestamp("updated_at", { withTimezone: true })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    uniqueIndex("api_keys_user_provider_idx").on(table.userId, table.provider),
    index("api_keys_user_id_idx").on(table.userId),
  ],
);

// ====================
// Sessions
// ====================
// Provider/model/variant live here because they describe the configuration
// of a conversation, not a permanent user setting.

export const sessions = pgTable(
  "sessions",
  {
    id: uuid("id").defaultRandom().primaryKey(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    // "telegram", "web", etc.
    source: text("source").notNull(),

    modelProvider: text("model_provider"),
    modelName: text("model_name"),
    modelVariant: text("model_variant"),

    // Complete PydanticAI message/event representation for this session.
    pydanticMessage: jsonb("pydantic_message").default([]).notNull(),

    isActive: boolean("is_active").default(true).notNull(),

    createdAt: timestamp("created_at", { withTimezone: true })
      .defaultNow()
      .notNull(),

    updatedAt: timestamp("updated_at", { withTimezone: true })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    index("sessions_user_id_idx").on(table.userId),
    // Enforces "only one active session per user" at the DB level.
    // Partial unique index: only rows where is_active = true participate.
    uniqueIndex("sessions_one_active_per_user_idx")
      .on(table.userId)
      .where(sql`is_active = true`),
  ],
);

// ====================
// Messages
// ====================

export const messages = pgTable(
  "messages",
  {
    id: uuid("id").defaultRandom().primaryKey(),

    sessionId: uuid("session_id")
      .notNull()
      .references(() => sessions.id, { onDelete: "cascade" }),

    // user / assistant / system / tool
    role: text("role").notNull(),
    content: text("content"),
    metadata: jsonb("metadata"),

    createdAt: timestamp("created_at", { withTimezone: true })
      .defaultNow()
      .notNull(),
  },
  (table) => [index("messages_session_id_idx").on(table.sessionId)],
);
