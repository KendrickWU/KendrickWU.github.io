---
layout: archive
permalink: /blog/
title: "Blog posts"
author_profile: true
redirect_from:
  - /wordpress/blog-posts/
---

{% include base_path %}

{% assign blog_post_count = 0 %}
{% capture written_year %}'None'{% endcapture %}

{% for post in site.posts %}
  {% assign is_writing_post = false %}
  {% if post.categories contains 'writing' or post.tags contains 'writing' %}
    {% assign is_writing_post = true %}
  {% endif %}

  {% unless is_writing_post %}
    {% assign blog_post_count = blog_post_count | plus: 1 %}
    {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
    {% if year != written_year %}
      <h2 id="{{ year | slugify }}" class="archive__subtitle">{{ year }}</h2>
      {% capture written_year %}{{ year }}{% endcapture %}
    {% endif %}
    {% include archive-single.html %}
  {% endunless %}
{% endfor %}

{% if blog_post_count == 0 %}
  <p>No blog posts yet.</p>
{% endif %}
