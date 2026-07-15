---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
redirect_from:
  - /publications/airline-cargo/
---

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

Current manuscripts are grouped by research program, so related conference and journal versions appear together instead of being counted as separate projects. The list below contains the five active research programs and their current submission status.

{% assign current_publications = site.publications | sort: "order" %}
{% for post in current_publications %}
  {% include archive-single.html %}
{% endfor %}
