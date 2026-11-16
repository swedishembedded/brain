-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- The book's EPUB (scripts/book/epub.py): what the HTML of an e-book needs
-- that the markdown does not say.
--
-- Figures are numbered within their chapter, "Figure 1", "Figure 2", as the
-- text cites them; the count restarts at every chapter and every top-level
-- page. A figure's caption is its alt text, unless the paragraph under the
-- image is a caption of the chapter's own ("**Figure 3.** ..."): that
-- paragraph becomes the caption, with its number, and the alt text stays the
-- image's description. The author's portrait (class "portrait") is not a
-- figure the text cites and stays unnumbered.
--
-- A table sits in a block that scrolls sideways, so a table wider than the
-- reader's screen scrolls instead of running off the page or being shrunk
-- unreadably small by the reading system.
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
-- The number of a paragraph that opens with a bold "Figure N.", or nil.
local function caption_number(block)
  if not block or block.t ~= "Para" or #block.content == 0 or block.content[1].t ~= "Strong" then
    return nil
  end
  return tonumber(pandoc.utils.stringify(block.content[1]):match("^Figure (%d+)%.$"))
end

-- A figure followed by its own caption paragraph takes that paragraph as
-- its caption; the number goes on the figure as attribute "number".
local function adopt_captions(blocks)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local b, n = blocks[i], caption_number(blocks[i + 1])
    if b.t == "Figure" and n then
      b.caption.long = pandoc.Blocks({ pandoc.Plain(blocks[i + 1].content) })
      b.attributes.number = tostring(n)
      out:insert(b)
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  return out
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
  { Blocks = adopt_captions },
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
        return fig
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
      return pandoc.Div({ t }, pandoc.Attr("", { "table-scroll" })), false
    end,
  },
}
