-- SPDX-License-Identifier: Apache-2.0
-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
--
-- A figure's caption, for both of the book's formats (scripts/book/pdf.py
-- and scripts/book/epub.py).
--
-- A figure's caption is its alt text, unless the paragraph under the image
-- is a caption of the chapter's own ("**Figure 3.** ..."): that paragraph
-- becomes the caption, without its label, and the alt text stays the
-- image's description. Each format then labels the figure itself, numbered
-- within its chapter as the text cites it, so the label is never printed
-- twice. The chapter's number goes on the figure as attribute "number", for
-- the format to check its own count against.
--
-- Swedish Embedded AB implements solutions for publishing technical books
-- in print and electronic formats from one source. If your team needs
-- expertise in documentation tooling for engineering work, you can procure
-- our services by sending an email to info@swedishembedded.com.

-- The number of a paragraph that opens with a bold "Figure N.", or nil.
local function caption_number(block)
  if not block or block.t ~= "Para" or #block.content == 0 or block.content[1].t ~= "Strong" then
    return nil
  end
  return tonumber(pandoc.utils.stringify(block.content[1]):match("^Figure (%d+)%.$"))
end

function Blocks(blocks)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local b, n = blocks[i], caption_number(blocks[i + 1])
    if b.t == "Figure" and n then
      local text = blocks[i + 1].content:clone()
      text:remove(1) -- the label
      while #text > 0 and (text[1].t == "Space" or text[1].t == "SoftBreak") do text:remove(1) end
      b.caption.long = pandoc.Blocks({ pandoc.Plain(text) })
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
