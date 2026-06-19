---
layout: single
title: "Writing"
permalink: /blog/
author_profile: true
header:
  overlay_color: "#3a3a3a"
  overlay_filter: "0.35"
  overlay_image: /images/blog/yokosuma-still-walking/05.jpg
  caption: "久里浜霊園 — 『歩いても歩いても』ロケ地"
excerpt: "「人生は、いつもちょっとだけ間に合わない。」<br/>—— 是枝裕和『歩いても歩いても』"
---

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'blog' or post.tags contains 'blog'" %}
{% if posts.size == 0 %}
  {% assign posts = site.posts %}
{% endif %}

<style>
  /* ===== Blog Card Feed ===== */
  .blog-card-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2.5em;
    margin-top: 2em;
  }

  .blog-card {
    background: #fff;
    border-radius: 2px;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .blog-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    transform: translateY(-2px);
  }

  /* Thumbnail */
  .blog-card__image {
    display: block;
    width: 100%;
    height: 260px;
    object-fit: cover;
    background: #f0f0f0;
  }
  .blog-card__image-link {
    display: block;
    overflow: hidden;
  }

  /* Body */
  .blog-card__body {
    padding: 1.5em 1.8em 1.8em;
  }

  /* Date */
  .blog-card__date {
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a0a0a0;
    margin-bottom: 0.5em;
    font-weight: 400;
  }

  /* Title */
  .blog-card__title {
    margin: 0 0 0.4em 0;
    font-size: 1.25em;
    font-weight: 500;
    line-height: 1.4;
  }
  .blog-card__title a {
    color: #2c2c2c;
    text-decoration: none;
  }
  .blog-card__title a:hover {
    color: #2f7f93;
    text-decoration: underline;
  }

  /* Meta */
  .blog-card__meta {
    font-size: 0.78em;
    color: #999;
    margin-bottom: 0.8em;
  }
  .blog-card__meta span {
    margin-right: 1em;
  }

  /* Excerpt */
  .blog-card__excerpt {
    font-size: 0.88em;
    color: #666;
    line-height: 1.7;
    margin-bottom: 1em;
  }

  /* Tags */
  .blog-card__tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
  }
  .blog-card__tag {
    display: inline-block;
    font-size: 0.7em;
    color: #888;
    text-decoration: none;
    padding: 3px 10px;
    border: 1px solid #e0e0e0;
    border-radius: 3px;
    transition: all 0.2s ease;
  }
  .blog-card__tag:hover {
    color: #2f7f93;
    border-color: #2f7f93;
    background: rgba(47,127,147,0.04);
  }

  /* Empty state */
  .blog-empty {
    text-align: center;
    color: #999;
    padding: 4em 0;
    font-size: 0.95em;
  }

  /* Desktop: wider cards */
  @media (min-width: 768px) {
    .blog-card__image {
      height: 320px;
    }
    .blog-card__body {
      padding: 1.8em 2.2em 2em;
    }
  }
</style>

<div class="blog-card-grid">
  {% for post in posts %}
    {% comment %} Extract thumbnail from post frontmatter {% endcomment %}
    {% assign thumb = post.header.overlay_image | default: post.header.teaser | default: post.header.image %}

    <article class="blog-card">
      {% if thumb %}
        <a href="{{ post.url }}" class="blog-card__image-link" aria-hidden="true">
          <img
            src="{{ thumb | relative_url }}"
            alt="{{ post.title | escape }}"
            class="blog-card__image"
            loading="lazy"
          >
        </a>
      {% endif %}

      <div class="blog-card__body">
        <div class="blog-card__date">{{ post.date | date: "%b %d, %Y" }}</div>

        <h2 class="blog-card__title">
          <a href="{{ post.url }}">{{ post.title }}</a>
        </h2>

        <div class="blog-card__meta">
          {% if post.read_time %}
            <span><i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span>
          {% endif %}
          {% if post.categories.size > 0 %}
            <span>
              <i class="fa fa-folder-open" aria-hidden="true"></i>
              {% for cat in post.categories %}
                {% unless cat == 'blog' %}
                  {{ cat }}{% unless forloop.last %}, {% endunless %}
                {% endunless %}
              {% endfor %}
            </span>
          {% endif %}
        </div>

        {% if post.excerpt %}
          <p class="blog-card__excerpt">
            {{ post.excerpt | strip_html | truncatewords: 45 }}
          </p>
        {% endif %}

        {% if post.tags.size > 0 %}
          <div class="blog-card__tags">
            {% for tag in post.tags %}
              {% unless tag == 'blog' %}
                <a href="/tags/{{ tag | slugify }}/" class="blog-card__tag">{{ tag }}</a>
              {% endunless %}
            {% endfor %}
          </div>
        {% endif %}
      </div>
    </article>
  {% endfor %}
</div>

{% if posts.size == 0 %}
  <p class="blog-empty">まだ記事がありません。お楽しみに。</p>
{% endif %}
