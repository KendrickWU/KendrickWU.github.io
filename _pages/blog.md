---
layout: single
title: "随筆"
permalink: /blog/
author_profile: true
header:
  overlay_color: "#5e616c"
  overlay_filter: "0.3"
  overlay_image: /images/blog/yokosuma-still-walking/05.jpg
  caption: "久里浜霊園 — 『歩いても歩いても』ロケ地"
excerpt: "「人生は、いつもちょっとだけ間に合わない。」<br/>—— 是枝裕和『歩いても歩いても』"
---

<style>
  .blog-feed {
    margin-top: 2em;
  }
  .blog-entry {
    display: flex;
    gap: 1.5em;
    padding: 2em 0;
    border-bottom: 1px solid #e8e8e8;
    align-items: flex-start;
  }
  .blog-entry:first-child {
    border-top: 1px solid #e8e8e8;
  }
  .blog-entry-date {
    flex: 0 0 80px;
    text-align: center;
    line-height: 1.2;
  }
  .blog-entry-date .day {
    display: block;
    font-size: 1.8em;
    font-weight: 300;
    color: #494e52;
  }
  .blog-entry-date .month-year {
    display: block;
    font-size: 0.75em;
    text-transform: uppercase;
    color: #898c8f;
    letter-spacing: 0.05em;
  }
  .blog-entry-content {
    flex: 1;
  }
  .blog-entry-content h2 {
    margin: 0 0 0.25em 0;
    font-size: 1.3em;
    font-weight: 500;
  }
  .blog-entry-content h2 a {
    color: #494e52;
    text-decoration: none;
  }
  .blog-entry-content h2 a:hover {
    color: #2f7f93;
    text-decoration: underline;
  }
  .blog-entry-meta {
    font-size: 0.8em;
    color: #898c8f;
    margin-bottom: 0.5em;
  }
  .blog-entry-meta span {
    margin-right: 1em;
  }
  .blog-entry-excerpt {
    font-size: 0.9em;
    color: #555;
    line-height: 1.6;
  }
  .blog-entry-tags {
    margin-top: 0.5em;
  }
  .blog-entry-tags a {
    display: inline-block;
    font-size: 0.7em;
    color: #898c8f;
    text-decoration: none;
    margin-right: 0.8em;
    padding: 2px 0;
    border-bottom: 1px dotted #ccc;
  }
  .blog-entry-tags a:hover {
    color: #2f7f93;
    border-bottom-color: #2f7f93;
  }
  @media (max-width: 600px) {
    .blog-entry {
      flex-direction: column;
      gap: 0.5em;
    }
    .blog-entry-date {
      flex: none;
      text-align: left;
    }
    .blog-entry-date .day {
      display: inline;
      font-size: 1.2em;
    }
    .blog-entry-date .month-year {
      display: inline;
      margin-left: 0.3em;
    }
  }
</style>

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'blog' or post.tags contains 'blog'" %}
{% if posts.size == 0 %}
  {% assign posts = site.posts %}
{% endif %}

<div class="blog-feed">
  {% for post in posts %}
  <div class="blog-entry">
    <div class="blog-entry-date">
      <span class="day">{{ post.date | date: "%d" }}</span>
      <span class="month-year">{{ post.date | date: "%b %Y" }}</span>
    </div>
    <div class="blog-entry-content">
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <div class="blog-entry-meta">
        {% if post.read_time %}
          <span><i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span>
        {% endif %}
        {% if post.categories.size > 0 %}
          <span><i class="fa fa-folder-open" aria-hidden="true"></i> {{ post.categories | join: ", " }}</span>
        {% endif %}
      </div>
      {% if post.excerpt %}
        <div class="blog-entry-excerpt">
          {{ post.excerpt | strip_html | truncatewords: 50 }}
        </div>
      {% endif %}
      {% if post.tags.size > 0 %}
        <div class="blog-entry-tags">
          <i class="fa fa-tags" aria-hidden="true" style="font-size:0.7em; color:#ccc; margin-right:0.3em;"></i>
          {% for tag in post.tags %}
            <a href="/tags/{{ tag | slugify }}/">{{ tag }}</a>
          {% endfor %}
        </div>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>

{% if posts.size == 0 %}
<p style="text-align: center; color: #898c8f; padding: 3em 0;">
  まだ記事がありません。お楽しみに。
</p>
{% endif %}
