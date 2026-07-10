# Wu Hongfan Homepage

Personal academic homepage built with Jekyll and Academic Pages. The site has two publishing surfaces: research-oriented posts use the main theme, while personal writing uses dedicated layouts and styles under `/writing/`.

## Common Commands

Install Ruby dependencies:

```sh
bundle install
```

Build the site:

```sh
bundle exec jekyll build
```

Validate generated pages, local links, asset references, document structure, and writing metadata:

```sh
bundle exec ruby script/validate_site.rb
```

Preview locally:

```sh
bundle exec jekyll serve --config _config.yml,_config.dev.yml -l -H 127.0.0.1
```

The local preview runs at `http://127.0.0.1:4000`. The development override keeps assets local and disables analytics. Changes to either config file require restarting the preview server.

## Content Model

Technical posts live in `_posts/` and normally use `layout: post`. That compatibility layout inherits the site's full `single` article layout. Set `mathjax: true` only on posts that contain equations.

Personal writing also lives in `_posts/`, with `layout: writing-post` and a `writing_type` matching a key in `_data/writing_categories.yml`. Category labels, descriptions, routes, icons, and empty states are maintained in that data file.

Writing series can add `writing_series`, `writing_order`, and `title_lines`. A local `source_path` is optional; when the source exists, the validator checks that all non-image body lines remain exactly unchanged.

## Project Structure

- `_config.yml`: site-wide settings, collections, plugins, and build excludes.
- `_pages/`: standalone pages such as About, CV, Publications, Blog, and Writing.
- `_posts/`: dated blog posts.
- `_publications/`, `_teaching/`, `_talks/`: structured academic collections.
- `_layouts/`, `_includes/`, `_sass/`, `assets/`: layouts, shared components, and styling code.
- `images/`, `files/`, `slides/`: public static assets.
- `markdown_generator/`: local helper scripts; excluded from the published site.
- `script/validate_site.rb`: post-build structural and content checks.

## Publishing Notes

Keep private notes, teaching materials, drafts, and generated build output outside the published site. The Jekyll `exclude` list in `_config.yml` and `.gitignore` both need to know about local-only directories.

Before pushing meaningful changes, run:

```sh
bundle exec jekyll build
bundle exec ruby script/validate_site.rb
git status --short
```
