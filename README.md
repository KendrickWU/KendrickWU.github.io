# Wu Hongfan Homepage

Personal academic homepage built with Jekyll and the Academic Pages theme.

## Common Commands

Install Ruby dependencies:

```sh
bundle install
```

Build the site:

```sh
bundle exec jekyll build
```

Preview locally:

```sh
bundle exec jekyll serve -l -H localhost
```

The local preview runs at `http://localhost:4000`.

## Project Structure

- `_config.yml`: site-wide settings, collections, plugins, and build excludes.
- `_pages/`: standalone pages such as About, CV, Publications, Blog, and Writing.
- `_posts/`: dated blog posts.
- `_publications/`, `_teaching/`, `_talks/`: structured academic collections.
- `_layouts/`, `_includes/`, `_sass/`, `assets/`: theme and styling code.
- `images/`, `files/`, `slides/`: public static assets.
- `markdown_generator/`: local helper scripts; excluded from the published site.

## Publishing Notes

Keep private notes, teaching materials, drafts, and generated build output outside the published site. The Jekyll `exclude` list in `_config.yml` and `.gitignore` both need to know about local-only directories.

Before pushing meaningful changes, run:

```sh
bundle exec jekyll build
git status --short
```
