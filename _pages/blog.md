---
layout: single
title: "Blog"
permalink: /blog/
author_profile: true
header:
  overlay_color: "#2f3f44"
  overlay_filter: "0.25"
excerpt: "旅行、电影、城市和一些慢慢写下来的个人观察。"
---

{% assign blog_post_count = 0 %}

<style>
  .blog-intro {
    margin: 0 0 2.2em;
    padding-bottom: 1.6em;
    border-bottom: 1px solid #e6e1da;
    color: #6e6a63;
    line-height: 1.9;
  }
  .blog-intro p {
    margin: 0.5em 0 0;
  }
  .blog-card-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5em;
    margin-top: 1.5em;
  }

  .blog-card {
    background: #fffdf9;
    border: 1px solid #ece5da;
    border-radius: 4px;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
    box-shadow: 0 1px 4px rgba(57, 47, 35, 0.05);
  }
  .blog-card:hover {
    box-shadow: 0 8px 24px rgba(57, 47, 35, 0.10);
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
    padding: 1.4em 1.5em 1.5em;
  }

  /* Date */
  .blog-card__date {
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9a8f7f;
    margin-bottom: 0.5em;
    font-weight: 400;
  }

  /* Title */
  .blog-card__title {
    margin: 0 0 0.4em 0;
    font-size: 1.28em;
    font-weight: 500;
    line-height: 1.4;
  }
  .blog-card__title a {
    color: #2d302c;
    text-decoration: none;
  }
  .blog-card__title a:hover {
    color: #287477;
    text-decoration: underline;
  }

  /* Meta */
  .blog-card__meta {
    font-size: 0.78em;
    color: #8f877b;
    margin-bottom: 0.8em;
  }
  .blog-card__meta span {
    margin-right: 1em;
  }

  /* Excerpt */
  .blog-card__excerpt {
    font-size: 0.88em;
    color: #665f55;
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
    color: #756b5d;
    text-decoration: none;
    padding: 3px 10px;
    border: 1px solid #e4dbcf;
    border-radius: 3px;
    transition: all 0.2s ease;
  }
  .blog-card__tag:hover {
    color: #287477;
    border-color: #287477;
    background: rgba(40,116,119,0.05);
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
      padding: 1.65em 1.9em 1.8em;
    }
  }
</style>

<div class="blog-intro">
  <p>这里会放一些旅行计划、路线整理、电影地点巡礼和个人感悟。希望它更像一间慢慢更新的小书房，而不是正式论文之外的另一个公告栏。</p>
  <p>第一篇从是枝裕和的《步履不停》开始：把横须贺、三浦半岛和电影里的几个地点串成一条可走的路线。</p>
</div>

<div class="blog-card-grid">
  {% for post in site.posts %}
    {% if post.categories contains 'blog' or post.tags contains 'blog' %}
      {% assign blog_post_count = blog_post_count | plus: 1 %}
      {% assign post_url = post.link | default: post.url %}
      {% comment %} Extract thumbnail from post frontmatter {% endcomment %}
      {% assign thumb = post.header.overlay_image | default: post.header.teaser | default: post.header.image %}

      <article class="blog-card">
        {% if thumb %}
          <a href="{{ post_url | relative_url }}" class="blog-card__image-link" aria-hidden="true">
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
            <a href="{{ post_url | relative_url }}">{{ post.title }}</a>
          </h2>

          <div class="blog-card__meta">
            {% if post.read_time %}
              <span><i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span>
            {% endif %}
            {% if post.categories.size > 0 %}
              <span>
                <i class="fa fa-folder-open" aria-hidden="true"></i>
                {% assign printed_category = false %}
                {% for cat in post.categories %}
                  {% unless cat == 'blog' %}
                    {% if printed_category %}, {% endif %}{{ cat }}
                    {% assign printed_category = true %}
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
                  {% assign tag_slug = tag | slugify %}
                  <a href="{{ '/tags/' | append: tag_slug | append: '/' | relative_url }}" class="blog-card__tag">{{ tag }}</a>
                {% endunless %}
              {% endfor %}
            </div>
          {% endif %}
        </div>
      </article>
    {% endif %}
  {% endfor %}
</div>

{% if blog_post_count == 0 %}
  <p class="blog-empty">まだ記事がありません。お楽しみに。</p>
{% endif %}
