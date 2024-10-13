function Link(el)
    -- Check if the link's content is exclusively images
    local allImages = true
    for i, content in ipairs(el.content) do
        if content.t ~= 'Image' then
            allImages = false
            break
        end
    end

    -- If the link contains only images, return those images directly
    if allImages then
        return el.content
    else
        -- If the link contains text or other types of content, convert to plain text
        local text = pandoc.utils.stringify(el.content)
        return pandoc.Str(text)
    end
end
