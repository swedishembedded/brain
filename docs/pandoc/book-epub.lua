-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- The book's EPUB (scripts/book/epub.py): what the HTML of an e-book needs
-- that the markdown does not say.
--
-- Figures are numbered within their chapter, "Figure 1", "Figure 2", as the
-- text cites them; the count restarts at every chapter and every top-level
-- page (the caption itself is book-figures.lua's, which runs first). A
-- caption whose own number differs from the count is reported. The author's
-- portrait (class "portrait") is not a figure the text cites and stays
-- unnumbered.
--
-- A table is set smaller when even its narrowest layout (each column as
-- wide as its longest word or number, which cannot wrap) is wider than a
-- phone's or a tablet column's line, down to 60% of its size; one still too
-- wide at that size may break its cells inside a word. It is never
-- put in a box that scrolls: a reading system that pages the book cannot
-- split a scrolling box across pages, and some draw nothing of the part
-- that falls on the next page, leaving the page blank above what follows.
--
-- Raw HTML in the source is an error: the book's markdown carries none, so
-- a `<word>` that markdown took for a tag is text the reader would lose (the
-- PDF drops it) or a page that is not well-formed; each one is reported
-- ("book-epub: raw HTML ...").
--
-- Last, it reports how many formulas, tables and images the book has, as
-- pandoc read them ("book-epub: counts ..." on standard error), so the build
-- can check that each of them is in the EPUB it wrote.
--
-- Swedish Embedded AB implements solutions for publishing technical books
-- in print and electronic formats from one source. If your team needs
-- expertise in documentation tooling for engineering work, you can procure
-- our services by sending an email to info@swedishembedded.com.

local figure = 0

local function restarts(header)
  return header.level == 1 or header.classes:includes("chapter")
end

local function portrait(fig)
  local is = false
  fig.content:walk({ Image = function(img)
    if img.classes:includes("portrait") then is = true end
  end })
  return is
end

-- Headers and figures must be seen in document order, so one top-down
-- traversal handles both. A table returns `false` as well, so the traversal
-- does not descend into the block that now holds it and wrap it again.
-- The widest a table may be at its narrowest, in ems of the table's own
-- (smaller) type: a 340 px phone line holds about 23 of them.
local TABLE_LIMIT_EM = 21
local TABLE_MIN_SCALE = 0.6

-- The width of a word in ems, by character class (digits about half an em,
-- capitals wider, lowercase about half).
local function word_em(word)
  local w = 0
  for _, cp in utf8.codes(word) do
    local c = utf8.char(cp)
    if c:match("%d") then w = w + 0.55
    elseif c:match("[A-Z]") then w = w + 0.7
    else w = w + 0.52 end
  end
  return w
end

-- A table's narrowest width: each column as wide as the longest word in
-- any of its cells, plus the cells' padding (0.6 em a side).
local function table_min_width(t)
  local cols = {}
  local function row(r)
    local c = 0
    for _, cell in ipairs(r.cells) do
      c = c + 1
      local longest = 0
      for word in pandoc.utils.stringify(cell.contents):gmatch("%S+") do
        longest = math.max(longest, word_em(word))
      end
      cols[c] = math.max(cols[c] or 0, longest)
    end
  end
  for _, r in ipairs(t.head.rows) do row(r) end
  for _, body in ipairs(t.bodies) do
    for _, r in ipairs(body.head) do row(r) end
    for _, r in ipairs(body.body) do row(r) end
  end
  local w = 0
  for _, c in pairs(cols) do w = w + c + 1.2 end
  return w
end

local counts = { Math = 0, Table = 0, Image = 0 }
local function count(el) counts[el.t] = counts[el.t] + 1 end

return {
  {
    Math = count,
    Table = count,
    Image = count,
    RawInline = function(r)
      if r.format:match("html") then io.stderr:write("book-epub: raw HTML " .. r.text .. "\n") end
    end,
    RawBlock = function(r)
      if r.format:match("html") then io.stderr:write("book-epub: raw HTML " .. r.text:sub(1, 80) .. "\n") end
    end,
    Pandoc = function()
      io.stderr:write(string.format("book-epub: counts math=%d table=%d image=%d\n",
        counts.Math, counts.Table, counts.Image))
    end,
  },
  {
    traverse = "topdown",
    Header = function(h)
      if restarts(h) then figure = 0 end
    end,
    Figure = function(fig)
      if portrait(fig) then return nil end
      figure = figure + 1
      if fig.attributes.number then
        if tonumber(fig.attributes.number) ~= figure then
          io.stderr:write(string.format("book-epub: figure captioned %s is figure %d of its chapter\n",
            fig.attributes.number, figure))
        end
        figure = tonumber(fig.attributes.number)
        fig.attributes.number = nil
      end
      local label = pandoc.Strong({ pandoc.Str("Figure"), pandoc.Space(), pandoc.Str(figure .. ".") })
      local caption = fig.caption.long
      if #caption > 0 and caption[1].t == "Plain" then
        caption[1].content:insert(1, pandoc.Space())
        caption[1].content:insert(1, label)
      else
        caption:insert(1, pandoc.Plain({ label }))
      end
      fig.caption.long = caption
      return fig
    end,
    Table = function(t)
      local width = table_min_width(t)
      if width <= TABLE_LIMIT_EM then return nil end
      local scale = math.max(TABLE_MIN_SCALE, TABLE_LIMIT_EM / width)
      -- A table that does not fit even at the smallest size may break its
      -- cells inside a word or number, so it still fits the line.
      local classes = { "table-fit" }
      if width * TABLE_MIN_SCALE > TABLE_LIMIT_EM then classes[2] = "table-tight" end
      return pandoc.Div({ t }, pandoc.Attr("", classes,
        { style = string.format("font-size: %d%%", math.floor(scale * 100)) })), false
    end,
  },
}
