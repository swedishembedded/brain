-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- Long display formulas on several lines, for the book's EPUB
-- (scripts/book/epub.py).
--
-- A printed page is one fixed width, so the PDF sets a display formula on
-- one line. An e-book page may be a phone or one column of a tablet, about
-- 22 ems wide, and MathML has no automatic line breaking that reading
-- systems implement: a formula wider than the column runs off the page. So
-- each display formula estimated wider than LIMIT_EM is broken the way a
-- typesetter would break it by hand:
--
--   1. independent formulas set side by side with \quad or \qquad go one
--      under the other;
--   2. a chain of relations (=, \le, \approx, \Rightarrow, ...) breaks before
--      a relation, the relations aligned under each other;
--   3. a side still too wide breaks before a binary + - \cdot or \times,
--      the continuation indented under the right-hand side; inside sized
--      delimiters only when nothing outside them is left, then after a
--      comma, and last between two factors of a product, the next row
--      starting with an explicit \times.
--
-- A formula is never broken inside braces, a \left...\right pair or an
-- environment. One the text already sets in rows (an aligned environment)
-- keeps its rows, and a row too wide is broken further after its alignment
-- point the same way; any other formula with rows is left as written.
-- The rows are substrings of the source, so nothing in the formula changes
-- but where its lines end.
--
-- A formula with no place left to break that is still too wide is set
-- smaller, down to half the text size, so it fits the column.
--
-- An equation number, \tag{N}, is set as "(N)" at the end of the formula's
-- last row: the MathML converter drops \tag, and the text cites equations
-- by their numbers.
--
-- Swedish Embedded AB implements solutions for publishing technical books
-- in print and electronic formats from one source. If your team needs
-- expertise in mathematical typesetting for engineering content, you can
-- procure our services by sending an email to info@swedishembedded.com.

-- The widest a row may be estimated to be, in ems of the text font. A
-- phone's column holds about 20 ems, a tablet's in two-column landscape
-- about 24, and the estimate is within 13% of what a browser sets for 98%
-- of the book's formulas (23% at worst), measured in Chromium in November
-- 2026.
local LIMIT_EM = 18
-- A comma inside parentheses is a break point only when the parenthesized
-- group is at least this wide: a long argument list, not an index [t, s].
local LONG_GROUP_EM = 12

-- ---- tokens ------------------------------------------------------------------

-- Each token: { kind = "cmd"|"char"|"open"|"close"|"space"|"script", text, s, e }
-- with s and e its byte range in the source.
local function tokenize(tex)
  local toks, i, n = {}, 1, #tex
  while i <= n do
    local c = tex:sub(i, i)
    local s = i
    if c == "\\" then
      local name = tex:match("^[A-Za-z]+", i + 1)
      if name then
        i = i + 1 + #name
      else
        i = i + 2
      end
      toks[#toks + 1] = { kind = "cmd", text = tex:sub(s + 1, i - 1), s = s, e = i - 1 }
    elseif c == "{" then
      toks[#toks + 1] = { kind = "open", text = c, s = s, e = s }
      i = i + 1
    elseif c == "}" then
      toks[#toks + 1] = { kind = "close", text = c, s = s, e = s }
      i = i + 1
    elseif c == "^" or c == "_" then
      toks[#toks + 1] = { kind = "script", text = c, s = s, e = s }
      i = i + 1
    elseif c:match("%s") then
      i = i + 1
      while i <= n and tex:sub(i, i):match("%s") do i = i + 1 end
      toks[#toks + 1] = { kind = "space", text = " ", s = s, e = i - 1 }
    else
      -- One character, whole if it is a multi-byte UTF-8 sequence.
      local len = 1
      local b = c:byte()
      if b >= 0xF0 then len = 4 elseif b >= 0xE0 then len = 3 elseif b >= 0xC0 then len = 2 end
      toks[#toks + 1] = { kind = "char", text = tex:sub(i, i + len - 1), s = s, e = i + len - 1 }
      i = i + len
    end
  end
  return toks
end

-- ---- width -------------------------------------------------------------------

-- Commands that take one argument whose width is the command's width.
local WRAP = {}
for w in ([[mathbf mathrm mathit mathsf mathtt mathcal mathbb mathfrak boldsymbol bm
  operatorname hat bar tilde vec dot ddot widehat widetilde overline underline
  mathring check breve acute grave mathop mathbin mathrel mathord displaystyle]]):gmatch("%S+") do
  WRAP[w] = true
end
local TEXT = { text = true, textrm = true, textit = true, textbf = true, texttt = true, mbox = true }
local SPACE = { [","] = 0.17, [":"] = 0.22, [";"] = 0.28, [" "] = 0.25, ["!"] = -0.17, quad = 1, qquad = 2 }
local DELIM = { left = true, right = true, big = true, Big = true, bigg = true, Bigg = true,
  bigl = true, bigr = true, Bigl = true, Bigr = true, biggl = true, biggr = true, Biggl = true,
  Biggr = true, middle = true }
local LARGE = { sum = 1.3, prod = 1.3, int = 0.9, oint = 0.9, bigcup = 1.2, bigcap = 1.2 }
-- Symbols set with space around them like a relation, but not places to
-- break a line.
local SPACED = { mid = true, to = true, mapsto = true, cdot = true, times = true, pm = true,
  mp = true, circ = true, odot = true, otimes = true, oplus = true, cup = true, cap = true,
  setminus = true, ast = true, star = true, dots = true, cdots = true, ldots = true }
-- Operators whose limits a display formula stacks under them, as words
-- (their own width), and braces that carry a label under or over them.
local STACKED = { max = 1.5, min = 1.4, lim = 1.2, sup = 1.3, inf = 1.2, argmax = 2.6, argmin = 2.5 }
local BRACE = { underbrace = true, overbrace = true }
local REL = {}
for w in ([[le leq ge geq ne neq approx equiv sim simeq cong propto Rightarrow Leftarrow
  Leftrightarrow iff implies impliedby ll gg prec succ preceq succeq in notin subset subseteq
  supset supseteq coloneqq]]):gmatch("%S+") do
  REL[w] = true
end

local width_from

-- The width of the item starting at token i (a group or one token), and the
-- index after it.
local function item(toks, i, j)
  while i <= j and toks[i].kind == "space" do i = i + 1 end
  if i > j then return 0, i end
  if toks[i].kind == "open" then
    local depth, k = 1, i + 1
    while k <= j and depth > 0 do
      if toks[k].kind == "open" then depth = depth + 1 elseif toks[k].kind == "close" then depth = depth - 1 end
      k = k + 1
    end
    return width_from(toks, i + 1, k - 2), k
  end
  return width_from(toks, i, i), i + 1
end

-- The text of a group's characters, for \text{...}.
local function text_width(toks, i, j)
  local _, k = item(toks, i, j)
  local w = 0
  for t = i, k - 1 do
    if toks[t].kind == "char" then w = w + 0.5 elseif toks[t].kind == "space" then w = w + 0.25 end
  end
  return w, k
end

-- Estimated width in ems of tokens i..j, set in display style. Generous
-- rather than tight: breaking a formula that would just have fitted costs
-- a line; one that overflows costs the reader the formula.
width_from = function(toks, i, j)
  local w = 0
  while i <= j do
    local t = toks[i]
    if t.kind == "space" or t.kind == "close" then
      i = i + 1
    elseif t.kind == "open" then
      local g
      g, i = item(toks, i, j)
      w = w + g
    elseif t.kind == "script" then
      local g
      g, i = item(toks, i + 1, j)
      w = w + 0.7 * g
    elseif t.kind == "char" then
      if t.text:match("[=<>+%-]") then
        w = w + 1.1
      elseif t.text:match("[,;]") then
        w = w + 0.45 -- with the space MathML sets after it
      elseif t.text == "." then
        w = w + 0.3
      elseif t.text:match("[()%[%]|/]") then
        w = w + 0.4
      else
        w = w + 0.6
      end
      i = i + 1
    else -- a command
      local name = t.text
      i = i + 1
      if name == "frac" or name == "dfrac" or name == "tfrac" or name == "binom" then
        local a, b
        a, i = item(toks, i, j)
        b, i = item(toks, i, j)
        w = w + math.max(a, b) * (name == "tfrac" and 0.7 or 0.95) + 0.3
      elseif name == "sqrt" then
        if toks[i] and toks[i].kind == "char" and toks[i].text == "[" then
          while i <= j and not (toks[i].kind == "char" and toks[i].text == "]") do i = i + 1 end
          i = i + 1
        end
        local a
        a, i = item(toks, i, j)
        w = w + a + 0.9
      elseif WRAP[name] then
        local a
        a, i = item(toks, i, j)
        -- Bold letters are wider than the italic ones the count assumes.
        w = w + a * ((name == "mathbf" or name == "boldsymbol" or name == "bm") and 1.3 or 1)
      elseif TEXT[name] then
        local a
        a, i = text_width(toks, i, j)
        w = w + a
      elseif SPACE[name] then
        w = w + SPACE[name]
      elseif DELIM[name] then
        local a
        a, i = item(toks, i, j)
        w = w + 0.1 + math.min(a, 0.5)
      elseif LARGE[name] or STACKED[name] or BRACE[name] then
        -- In a display formula the limits of a large operator (and of max,
        -- min, lim, ...) and the label of an under- or overbrace sit above
        -- or below it: the width is the wider of the two, not their sum.
        local base
        if BRACE[name] then
          base, i = item(toks, i, j)
        else
          base = LARGE[name] or STACKED[name]
        end
        local scripts = 0
        while true do
          local k = i
          while k <= j and toks[k].kind == "space" do k = k + 1 end
          if k > j or toks[k].kind ~= "script" then break end
          local a
          a, i = item(toks, k + 1, j)
          scripts = math.max(scripts, 0.7 * a)
        end
        w = w + math.max(base, scripts)
      elseif REL[name] or SPACED[name] then
        w = w + 1.3
      elseif name == "begin" then
        -- An environment is as wide as its widest row, a row as wide as its
        -- cells together with the space between columns; its delimiters
        -- (a matrix's brackets, the brace of cases) add about an em.
        local _
        _, i = item(toks, i, j)
        local depth, k = 1, i
        local rows, cells, cell_start, widest = 0, 0, i, 0
        local row_width = 0
        local function close_cell(e)
          row_width = row_width + width_from(toks, cell_start, e)
        end
        while k <= j and depth > 0 do
          local t2 = toks[k]
          if t2.kind == "cmd" and t2.text == "begin" then
            depth = depth + 1
          elseif t2.kind == "cmd" and t2.text == "end" then
            depth = depth - 1
            if depth == 0 then break end
          elseif depth == 1 and t2.kind == "char" and t2.text == "&" then
            close_cell(k - 1)
            row_width = row_width + 1
            cell_start = k + 1
          elseif depth == 1 and t2.kind == "cmd" and t2.text == "\\" then
            close_cell(k - 1)
            widest = math.max(widest, row_width)
            row_width, cell_start = 0, k + 1
          end
          k = k + 1
        end
        close_cell(k - 1)
        widest = math.max(widest, row_width)
        w = w + widest + 1
        -- Past the \end and its name.
        _, i = item(toks, k + 1, j)
      elseif name == "end" then
        local _
        _, i = item(toks, i, j)
      elseif name:match("^[A-Za-z]+$") then
        w = w + (#name > 3 and 1.0 or 0.65) -- a Greek letter or symbol; ln, max, log are words
        if name == "ln" or name == "log" or name == "exp" or name == "max" or name == "min"
            or name == "sin" or name == "cos" or name == "tan" or name == "tanh" or name == "arg" then
          w = w + 0.4
        end
      else
        w = w + 0.3
      end
    end
  end
  return w
end

-- ---- breaking ----------------------------------------------------------------

local OPEN_DELIM = { ["("] = true, ["["] = true, lbrace = true, ["{"] = true, langle = true,
  lfloor = true, lceil = true, lVert = true, lvert = true }
local CLOSE_DELIM = { [")"] = true, ["]"] = true, rbrace = true, ["}"] = true, rangle = true,
  rfloor = true, rceil = true, rVert = true, rvert = true }
local ARROW = { Rightarrow = true, Leftarrow = true, Leftrightarrow = true, iff = true, implies = true,
  impliedby = true }

-- For every token: whether a line may break before it, how deep it is
-- inside sized delimiters (\bigl( ... \bigr), \big[ ... \big]) and how deep
-- inside plain parentheses and brackets. A line never breaks inside braces,
-- a \left...\right pair or an environment, which TeX and MathML need on one
-- row. A typesetter breaks inside sized delimiters, which enclose long
-- expressions, before inside plain ones, which enclose short ones such as
-- (s - 2); inside plain ones only after a comma. An unbalanced bracket, as in
-- the interval [0, 1), only makes fewer breaks possible.
local function nesting(toks)
  local hard, sized, plain = 0, 0, 0
  local breakable, sizeds, plains = {}, {}, {}
  local i = 1
  while i <= #toks do
    local t = toks[i]
    local opens = t.kind == "open" or (t.kind == "cmd" and (t.text == "left" or t.text == "begin"))
    local closes = t.kind == "close" or (t.kind == "cmd" and (t.text == "right" or t.text == "end"))
    if closes then hard = math.max(0, hard - 1) end
    breakable[i] = hard == 0 and not opens and not closes
    sizeds[i], plains[i] = sized, plain
    if breakable[i] then
      if t.kind == "cmd" and DELIM[t.text] then
        -- A sized delimiter and the delimiter it sizes are one token here.
        local d = toks[i + 1] and toks[i + 1].text or ""
        if t.text:match("l$") or (not t.text:match("r$") and OPEN_DELIM[d]) then
          sized = sized + 1
        elseif t.text:match("r$") or CLOSE_DELIM[d] then
          sized = math.max(0, sized - 1)
          sizeds[i] = sized
        end
        if toks[i + 1] then
          breakable[i + 1], sizeds[i + 1], plains[i + 1] = false, sized, plain
          i = i + 1
        end
      elseif t.kind == "char" and (t.text == "(" or t.text == "[") then
        plain = plain + 1
      elseif t.kind == "char" and (t.text == ")" or t.text == "]") then
        plain = math.max(0, plain - 1)
        plains[i] = plain
      end
    end
    if opens then hard = hard + 1 end
    i = i + 1
  end
  return breakable, sizeds, plains
end

local function is_relation(t)
  return (t.kind == "char" and (t.text == "=" or t.text == "<" or t.text == ">"))
    or (t.kind == "cmd" and REL[t.text] ~= nil)
end

-- Is the token before index i (skipping spaces) an operand, so that a + or
-- - at i is binary?
local function after_operand(toks, i, first)
  local k = i - 1
  while k >= first and toks[k].kind == "space" do k = k - 1 end
  if k < first then return false end
  local t = toks[k]
  if t.kind == "close" then return true end
  if t.kind == "char" then return not t.text:match("[=<>+%-(%[,&]") end
  if t.kind == "cmd" then
    return not (REL[t.text] or SPACE[t.text] or (DELIM[t.text] and t.text:match("l$"))
      or t.text == "pm" or t.text == "mp" or t.text == "cdot" or t.text == "times")
  end
  return false
end

local function src(tex, toks, a, b)
  return (tex:sub(toks[a].s, toks[b].e):gsub("^%s+", ""):gsub("%s+$", ""))
end

-- The break points of each level, loosest binding first: implication
-- arrows, other relations, binary + - \cdot \times outside parentheses, the
-- same inside sized delimiters, after a comma, and last between two factors
-- of a product (before a function such as \cos or \sum that follows a
-- closed group), where the row that continues starts with an explicit \times.
local LEVELS = 6
local BINARY = { pm = true, mp = true, cdot = true, times = true }
local FACTOR = { sin = true, cos = true, tan = true, exp = true, log = true, ln = true, sum = true,
  prod = true }

-- Is a line break before token i one between two factors of a product?
local function product_cut(toks, nest, i, a)
  local t = toks[i]
  if not (nest[1][i] and nest[3][i] == 0 and t.kind == "cmd" and FACTOR[t.text]) then return false end
  local k = i - 1
  while k >= a and toks[k].kind == "space" do k = k - 1 end
  if k < a then return false end
  local p = toks[k]
  return p.kind == "close" or (p.kind == "char" and (p.text == ")" or p.text == "]"))
    or (p.kind == "cmd" and (p.text == "rVert" or p.text == "rvert" or p.text == "rbrace"))
end

local function cuts(toks, nest, i, a, level)
  local breakable, sizeds, plains = nest[1], nest[2], nest[3]
  if not breakable[i] then return false end
  local t = toks[i]
  if level <= 4 and plains[i] > 0 then return false end
  if level <= 3 and sizeds[i] > 0 then return false end
  if level == 1 then return t.kind == "cmd" and ARROW[t.text] == true end
  if level == 2 then
    if t.kind == "char" and t.text == ":" and toks[i + 1] and toks[i + 1].text == "=" then return true end
    if not is_relation(t) then return false end
    local k = i - 1
    while k >= a and toks[k].kind == "space" do k = k - 1 end
    return not (toks[k] and toks[k].kind == "char" and toks[k].text == ":")
  end
  if level == 5 then
    if t.kind == "space" then return false end
    local k = i - 1
    while k >= a and toks[k].kind == "space" do k = k - 1 end
    if not (k >= a and toks[k].kind == "char" and toks[k].text == "," and breakable[k]) then return false end
    if plains[i] == 0 then return true end
    -- The group around the comma: from its opening to its closing bracket.
    local o, c = k, i
    while o > 1 and not (plains[o] == plains[i] - 1 and (toks[o].text == "(" or toks[o].text == "[")) do o = o - 1 end
    while c < #toks and not (plains[c] == plains[i] - 1 and (toks[c].text == ")" or toks[c].text == "]")) do c = c + 1 end
    return width_from(toks, o, c) >= LONG_GROUP_EM
  end
  if level == 6 then return product_cut(toks, nest, i, a) end
  return ((t.kind == "char" and (t.text == "+" or t.text == "-")) or (t.kind == "cmd" and BINARY[t.text] == true))
    and after_operand(toks, i, a)
end

-- Lines of token range [a, b]: unbroken when it fits, otherwise broken at
-- the loosest level that offers a break, each piece broken further as it
-- needs, and neighbouring pieces joined again while they fit.
local continuation

local function lines(toks, nest, a, b, level)
  if level > LEVELS or width_from(toks, a, b) <= LIMIT_EM then return { { a, b } } end
  local starts = { a }
  for i = a + 1, b do
    if cuts(toks, nest, i, a, level) then starts[#starts + 1] = i end
  end
  if #starts == 1 then return lines(toks, nest, a, b, level + 1) end
  local out = {}
  for n, s in ipairs(starts) do
    local e = (starts[n + 1] or b + 1) - 1
    for _, l in ipairs(lines(toks, nest, s, e, level + 1)) do
      local last = out[#out]
      if last and width_from(toks, last[1], l[2]) <= LIMIT_EM then
        last[2] = l[2]
      else
        out[#out + 1] = l
      end
    end
  end
  return out
end

-- The start of a row that continues an expression: indented under the
-- right-hand side, and with the \times a product break leaves implicit.
continuation = function(toks, nest, i, a)
  return product_cut(toks, nest, i, 1) and "&\\quad \\times " or "&\\quad "
end

local function first_token(toks, a, b)
  while a <= b and toks[a].kind == "space" do a = a + 1 end
  return toks[a]
end

-- One formula without \quad separators, as rows of an aligned block (and
-- true), or its source unchanged when it fits or offers no place to break.
local function break_chain(tex, toks, nest, a, b)
  local breakable, sizeds, plains = nest[1], nest[2], nest[3]
  local ls = lines(toks, nest, a, b, 1)
  if #ls < 2 then return src(tex, toks, a, b), false end
  -- Rows that start with a relation align on it, and the first row aligns
  -- on its own first relation. A row that continues an expression is
  -- indented under the right-hand side. When a long left side fills the
  -- first row alone, aligning the relations after it would make every row
  -- as wide as that side plus its own, so all rows start at the left margin
  -- instead and the relations are indented under the left side.
  local relation_rows = false
  for n = 2, #ls do
    if is_relation(first_token(toks, ls[n][1], ls[n][2])) then relation_rows = true end
  end
  local out = {}
  local left = false
  for n, l in ipairs(ls) do
    if left then
      local lead = is_relation(first_token(toks, l[1], l[2])) and "&\\quad " or continuation(toks, nest, l[1], b):gsub("^&\\quad", "&\\qquad")
      out[n] = lead .. src(tex, toks, l[1], l[2])
    elseif n == 1 then
      local cut
      for i = l[1] + 1, l[2] do
        if breakable[i] and sizeds[i] == 0 and plains[i] == 0 and is_relation(toks[i]) then cut = i break end
      end
      if cut then
        out[1] = src(tex, toks, l[1], cut - 1) .. " &" .. src(tex, toks, cut, l[2])
      elseif relation_rows and width_from(toks, l[1], l[2]) > LIMIT_EM / 3 then
        out[1] = "&" .. src(tex, toks, l[1], l[2])
        left = true
      elseif relation_rows then
        out[1] = src(tex, toks, l[1], l[2]) .. " &"
      else
        out[1] = "&" .. src(tex, toks, l[1], l[2])
      end
    elseif is_relation(first_token(toks, l[1], l[2])) then
      out[n] = "&" .. src(tex, toks, l[1], l[2])
    else
      out[n] = continuation(toks, nest, l[1], b) .. src(tex, toks, l[1], l[2])
    end
  end
  return "\\begin{aligned}" .. table.concat(out, " \\\\ ") .. "\\end{aligned}", true
end

local break_rows

-- `tex` broken into rows, with its \tag{N} (if any) as "(N)" at the end.
local function break_display(tex)
  local body, number = tex:match("^(.-)\\tag%s*(%b{})%s*$")
  if not body then return break_rows(tex) end
  local label = "\\qquad\\text{(" .. number:sub(2, -2) .. ")}"
  local broken = break_rows(body, 3)
  local stem, env = broken:match("^(.*)(\\end{%a+})$")
  if stem and broken ~= body then return stem .. " " .. label .. env end
  return body:gsub("%s+$", "") .. " " .. label
end

-- `tex` as rows when it is wider than LIMIT_EM, less `reserve` ems kept
-- free at the end for an equation number.
-- An aligned environment of the source, its long rows broken after their
-- alignment point; nil when `tex` is not one environment with one
-- alignment point per row.
local function break_aligned(tex)
  local lead, inner, trail = tex:match("^(%s*\\begin{aligned})(.*)(\\end{aligned}%s*[.,;]?%s*)$")
  if not inner or inner:find("\\begin{aligned}") then return nil end
  local toks = tokenize(inner)
  local nest = { nesting(toks) }
  local breakable = nest[1]
  -- Rows at the top level of the environment.
  local rows, start = {}, 1
  for i = 1, #toks do
    if breakable[i] and toks[i].kind == "cmd" and toks[i].text == "\\" then
      rows[#rows + 1] = { start, i - 1 }
      start = i + 1
    end
  end
  rows[#rows + 1] = { start, #toks }
  local out, changed = {}, false
  local function quad(i)
    return nest[1][i] and nest[2][i] == 0 and nest[3][i] == 0 and toks[i].kind == "cmd"
      and (toks[i].text == "quad" or toks[i].text == "qquad")
  end
  for _, r in ipairs(rows) do
    local amp
    for i = r[1], r[2] do
      if breakable[i] and toks[i].kind == "char" and toks[i].text == "&" then
        if amp then return nil end -- several alignment points: leave it
        amp = i
      end
    end
    if r[1] > r[2] or width_from(toks, r[1], r[2]) <= LIMIT_EM then
      out[#out + 1] = r[1] <= r[2] and inner:sub(toks[r[1]].s, toks[r[2]].e) or ""
    else
      changed = true
      -- Equations set side by side in the row each start a row of their own,
      -- aligned on their own first relation.
      local subs, start = {}, r[1]
      for i = (amp or r[1]) + 1, r[2] do
        if quad(i) then
          subs[#subs + 1] = { start, i - 1 }
          start = i + 1
        end
      end
      subs[#subs + 1] = { start, r[2] }
      for k, sub in ipairs(subs) do
        local head, a
        if k == 1 then
          head = amp and inner:sub(toks[sub[1]].s, toks[amp].e) or "&"
          a = (amp or sub[1] - 1) + 1
        else
          local rel
          for i = sub[1], sub[2] do
            if breakable[i] and nest[2][i] == 0 and nest[3][i] == 0 and is_relation(toks[i]) then rel = i break end
          end
          head = rel and rel > sub[1] and (src(inner, toks, sub[1], rel - 1) .. " &") or "&"
          a = rel or sub[1]
        end
        while a <= sub[2] and toks[a].kind == "space" do a = a + 1 end
        if a <= sub[2] then
          local ls = width_from(toks, sub[1], sub[2]) > LIMIT_EM and lines(toks, nest, a, sub[2], 2) or { { a, sub[2] } }
          out[#out + 1] = head .. src(inner, toks, ls[1][1], ls[1][2])
          for n = 2, #ls do
            local first = first_token(toks, ls[n][1], ls[n][2])
            out[#out + 1] = (is_relation(first) and "&" or continuation(toks, nest, ls[n][1], a))
              .. src(inner, toks, ls[n][1], ls[n][2])
          end
        else
          out[#out + 1] = head
        end
      end
    end
  end
  if not changed then return nil end
  return lead .. table.concat(out, " \\\\ ") .. trail
end

break_rows = function(tex, reserve)
  reserve = reserve or 0
  if tex:find("\\begin{aligned}") then return break_aligned(tex) or tex end
  if tex:find("\\\\") or tex:find("&") or tex:find("\\label") then return tex end
  local toks = tokenize(tex)
  if #toks == 0 or width_from(toks, 1, #toks) + reserve <= LIMIT_EM then return tex end
  local nest = { nesting(toks) }
  local breakable, sizeds, plains = nest[1], nest[2], nest[3]
  -- Independent formulas side by side go one under the other, as many to a
  -- row as fit.
  local parts, start = {}, 1
  for i = 2, #toks do
    if breakable[i] and sizeds[i] == 0 and plains[i] == 0 and toks[i].kind == "cmd"
        and (toks[i].text == "quad" or toks[i].text == "qquad") then
      parts[#parts + 1] = { start, i - 1 }
      start = i + 1
    end
  end
  parts[#parts + 1] = { start, #toks }
  local joined = {}
  for _, p in ipairs(parts) do
    local last = joined[#joined]
    if last and width_from(toks, last[1], p[2]) <= LIMIT_EM then
      last[2] = p[2]
    else
      joined[#joined + 1] = { p[1], p[2] }
    end
  end
  parts = joined
  local rows, changed = {}, #parts > 1
  for _, p in ipairs(parts) do
    local a = p[1]
    while a <= p[2] and toks[a].kind == "space" do a = a + 1 end
    if a <= p[2] then
      local row, broke = break_chain(tex, toks, nest, a, p[2])
      rows[#rows + 1] = row
      changed = changed or broke
    end
  end
  if not changed then return tex end
  if #rows == 1 then return rows[1] end
  return "\\begin{gathered}" .. table.concat(rows, " \\\\ ") .. "\\end{gathered}"
end

-- Smallest a formula is set, as a fraction of the text size: the widest
-- formula that cannot be broken (a boxed set of three equations) needs it
-- to fit a 340 px phone line.
local MIN_SCALE = 0.5
-- The width a formula that cannot be broken further is scaled to: below
-- LIMIT_EM by the estimate's error (it is within 13% of what a browser sets
-- for 98% of the formulas), so the scaled formula also fits a phone.
local SCALE_TARGET_EM = 15.3

-- The estimated width of `tex` as set: for an aligned or gathered block,
-- the widest left column plus the widest right column of its rows (the
-- columns of an aligned block add up), each row's width found the same way
-- when it is itself such a block; for anything else, its width on one line.
local function layout_width(tex)
  local env, inner, rest = tex:match("^%s*\\begin{(%a+)}(.*)\\end{%1}(.-)$")
  if env ~= "aligned" and env ~= "gathered" then
    local t = tokenize(tex)
    return width_from(t, 1, #t)
  end
  local toks = tokenize(inner)
  local breakable = nesting(toks)
  local rows, start = {}, 1
  for i = 1, #toks + 1 do
    if i > #toks or (breakable[i] and toks[i].kind == "cmd" and toks[i].text == "\\") then
      if i > start then rows[#rows + 1] = { start, i - 1 } end
      start = i + 1
    end
  end
  local left, right = 0, 0
  for _, r in ipairs(rows) do
    local amp
    for i = r[1], r[2] do
      if breakable[i] and toks[i].kind == "char" and toks[i].text == "&" then amp = i break end
    end
    local function part(a, b)
      if a > b then return 0 end
      return layout_width(inner:sub(toks[a].s, toks[b].e))
    end
    if amp then
      left = math.max(left, part(r[1], amp - 1))
      right = math.max(right, part(amp + 1, r[2]))
    else
      right = math.max(right, part(r[1], r[2]))
    end
  end
  local t = tokenize(rest)
  return left + right + width_from(t, 1, #t)
end

function Math(m)
  if m.mathtype ~= "DisplayMath" then return nil end
  m.text = break_display(m.text)
  -- A formula still too wide (one with no place to break, such as a long
  -- product of matrices or a boxed set of equations) is set smaller, to fit
  -- the column, rather than run past it.
  local widest = layout_width(m.text)
  if widest <= SCALE_TARGET_EM then return m end
  local scale = math.max(MIN_SCALE, SCALE_TARGET_EM / widest)
  return pandoc.Span({ m }, pandoc.Attr("", { "math-scaled" },
    { style = string.format("font-size: %d%%", math.floor(scale * 100)) }))
end

-- The estimator and the breaker, for the spec (tests/gates/book_epub.bats)
-- and for calibrating LIMIT_EM against a browser's measurements.
BookMathLines = {
  width = function(tex) local t = tokenize(tex) return width_from(t, 1, #t) end,
  break_display = break_display,
}
