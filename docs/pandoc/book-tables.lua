-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Column widths for the book's tables in the PDF.
--
-- A pipe table wider than pandoc's line limit gets column widths from the
-- dashes of its separator row. The book's tables are generated with three
-- dashes per column, so every column got the same narrow width and long
-- numbers ran into their neighbours or broke one character per line. This
-- filter sizes each column by what it has to hold instead: the longest body
-- cell, or the longest single word of its header (headers may wrap between
-- words). A table that does not fit the text width at the body size is set
-- smaller; one that still does not fit wraps the cells that have spaces to
-- wrap at (a number never breaks), and one too wide even then is set on a
-- page turned to landscape.
--
-- Swedish Embedded AB implements solutions for typesetting technical books
-- from generated sources. If your team needs expertise in documentation
-- tooling for engineering work, you can procure our services by sending an
-- email to info@swedishembedded.com.

-- Line widths of the book (docs/pandoc/book.tex: a 7in x 10in page with
-- 0.95in + 0.75in side margins and 0.95in top and bottom): the text width of a
-- page, and the text height, which is the line width of a page turned to
-- landscape for a table too wide to stand upright.
local PORTRAIT_PT = 5.3 * 72.27
local LANDSCAPE_PT = 8.1 * 72.27

-- Font sizes the table may be set at (body, then smaller), in points, and
-- the space LaTeX puts on each side of a column at that size.
local SIZES = {
  { name = nil, size_pt = 10, sep_pt = 6 },
  { name = "footnotesize", size_pt = 8, sep_pt = 5 },
  { name = "scriptsize", size_pt = 7, sep_pt = 3 },
}

-- Widths in ems, read off typeset pages with a little room to spare: P052's
-- figures and capitals are just over half an em, its lowercase about half,
-- a space a quarter; inline code is a
-- monospaced font whose every character measures about 0.67 em.
local function char_em(c)
  if c:match("[%dA-Z]") then return 0.55 end
  if c == " " then return 0.28 end
  return 0.5
end
local CODE_EM = 0.67

local function text_em(s)
  local w = 0
  for _, cp in utf8.codes(s) do
    w = w + char_em(utf8.char(cp))
  end
  return w
end

local function longest_word_em(s, code)
  local best = 0
  for word in s:gmatch("%S+") do
    best = math.max(best, code and (utf8.len(word) or #word) * CODE_EM or text_em(word))
  end
  return best
end

-- The width of a cell in ems: its whole text, and the narrowest it may be
-- set. A cell of prose (two or more words of three letters or more) may wrap
-- between words, so it needs only its longest word; a cell of tokens such as
-- "1 x 2560 x 2048" reads only on one line, except that a list of them may
-- break after a comma.
local function measure(cell)
  local code_whole, code_word = 0, 0
  pandoc.Blocks(cell.contents):walk({
    Code = function(c)
      code_whole = code_whole + (utf8.len(c.text) or #c.text) * CODE_EM
      code_word = math.max(code_word, longest_word_em(c.text, true))
    end,
  })
  -- The plain text without the code spans.
  local stripped = pandoc.Blocks(cell.contents):walk({
    Code = function() return pandoc.Str("") end,
  })
  local text = pandoc.utils.stringify(stripped)
  local whole = text_em(text) + code_whole
  local words = 0
  for _ in pandoc.utils.stringify(cell.contents):gmatch("%a%a%a+") do
    words = words + 1
  end
  if words < 2 then
    -- Tokens break only after a comma, as a vector "(0.74, 0.10, 0.11)" may.
    local chunk = 0
    for part in text:gmatch("[^,]+") do
      chunk = math.max(chunk, text_em(part:match("^%s*(.-)%s*$")) + 0.3)
    end
    if code_whole > 0 or chunk == 0 then
      return whole, whole
    end
    return whole, chunk
  end
  return whole, math.max(code_word, longest_word_em(text, false))
end

local function cell_text(cell)
  return pandoc.utils.stringify(cell.contents)
end

-- Widths (in ems) for each column at one size, or nil when even the
-- unbreakable words do not fit. A column gets its longest cell when the
-- table allows; otherwise every column keeps its longest word and the spare
-- space goes to the columns whose cells can wrap between words.
local function fit(full, word, avail_em)
  local n, sum_full, sum_word = #full, 0, 0
  for i = 1, n do
    sum_full = sum_full + full[i]
    sum_word = sum_word + word[i]
  end
  if sum_full <= avail_em then
    return full
  end
  if sum_word > avail_em then
    return nil
  end
  local spare, want = avail_em - sum_word, sum_full - sum_word
  local widths = {}
  for i = 1, n do
    widths[i] = word[i] + spare * (full[i] - word[i]) / want
  end
  return widths
end

function Table(tbl)
  local ncols = #tbl.colspecs
  local full, word = {}, {}
  for i = 1, ncols do
    full[i], word[i] = 1, 1
  end
  for _, row in ipairs(tbl.head.rows) do
    for i, cell in ipairs(row.cells) do
      if i <= ncols then
        local _, w = measure(cell)
        full[i] = math.max(full[i], w)
        word[i] = math.max(word[i], w)
      end
    end
  end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      for i, cell in ipairs(row.cells) do
        if i <= ncols then
          local whole, w = measure(cell)
          full[i] = math.max(full[i], whole)
          word[i] = math.max(word[i], w)
        end
      end
    end
  end

  local size, widths, line_pt
  for _, page_pt in ipairs({ PORTRAIT_PT, LANDSCAPE_PT }) do
    for _, s in ipairs(SIZES) do
      size, line_pt = s, page_pt
      widths = fit(full, word, (page_pt - ncols * 2 * s.sep_pt) / s.size_pt)
      if widths then break end
    end
    if widths then break end
  end
  if not widths then
    -- Wider than a landscape page even with every cell wrapped to one word:
    -- set it anyway and let LaTeX report the overflow.
    widths = word
    io.stderr:write("book-tables: a table with columns '" ..
      cell_text(tbl.head.rows[1].cells[1]) .. "' ... does not fit a landscape page\n")
  end
  -- pandoc sets a column to this fraction of the line width less the padding.
  local avail = line_pt - ncols * 2 * size.sep_pt
  for i = 1, ncols do
    tbl.colspecs[i] = { tbl.colspecs[i][1], widths[i] * size.size_pt / avail }
  end
  local before, after = "", ""
  if size.name then
    before = "\\begingroup\\" .. size.name .. string.format("\\setlength{\\tabcolsep}{%gpt}", size.sep_pt)
    after = "\\endgroup"
  end
  if line_pt == LANDSCAPE_PT then
    before, after = "\\begin{landscape}" .. before, after .. "\\end{landscape}"
  end
  if before == "" then
    return tbl
  end
  return { pandoc.RawBlock("latex", before), tbl, pandoc.RawBlock("latex", after) }
end
