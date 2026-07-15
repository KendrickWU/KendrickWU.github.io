#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "nokogiri"
require "pathname"
require "uri"
require "yaml"

ROOT = Pathname(__dir__).join("..").expand_path
SITE_ROOT = ROOT.join("_site")
CONFIG_PATH = ROOT.join("_config.yml")

def load_yaml(path)
  YAML.safe_load(
    path.read,
    permitted_classes: [Date, Time],
    aliases: true
  )
end

def read_front_matter(path)
  source = path.read
  match = source.match(/\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n/m)
  raise "missing YAML front matter" unless match

  metadata = YAML.safe_load(
    match[1],
    permitted_classes: [Date, Time],
    aliases: true
  ) || {}

  [metadata, match.post_match]
end

def preserved_text_lines(source)
  source.lines.map(&:chomp).reject do |line|
    line.strip.empty? || line.match?(/\A\s*!\[[^\]]*\]\(.*\)\s*\z/)
  end
end

def local_target(raw_url, source_file, site_root, site_host)
  return if raw_url.nil? || raw_url.empty?
  return if raw_url.start_with?("#", "mailto:", "tel:", "javascript:", "data:", "//")

  uri = URI.parse(raw_url)
  if uri.scheme
    return unless %w[http https].include?(uri.scheme)
    return unless uri.host&.downcase == site_host
  end

  path = uri.path.to_s
  return if path.empty?

  decoded_path = URI::DEFAULT_PARSER.unescape(path)
  if decoded_path.start_with?("/")
    site_root.join(decoded_path.sub(%r{\A/+}, ""))
  else
    source_file.dirname.join(decoded_path).cleanpath
  end
rescue URI::InvalidURIError, ArgumentError
  nil
end

def target_exists?(target)
  candidates = [target]
  candidates << target.join("index.html") if target.directory?
  if target.extname.empty?
    candidates << Pathname("#{target}.html")
    candidates << target.join("index.html")
  end
  candidates.uniq.any?(&:file?)
end

abort "Build output is missing. Run `bundle exec jekyll build` first." unless SITE_ROOT.directory?

config = load_yaml(CONFIG_PATH)
site_host = URI.parse(config.fetch("url")).host.downcase
errors = []

writing_categories = load_yaml(ROOT.join("_data/writing_categories.yml"))
writing_types = writing_categories.map { |category| category.fetch("key") }

writing_categories.each do |category|
  %w[key title url icon description empty].each do |field|
    errors << "_data/writing_categories.yml: #{category['key'] || 'category'} is missing #{field}" if category[field].to_s.strip.empty?
  end

  category_target = SITE_ROOT.join(category["url"].to_s.sub(%r{\A/+}, ""))
  errors << "_data/writing_categories.yml: missing generated route #{category['url']}" unless target_exists?(category_target)
end

writing_types.tally.each do |writing_type, count|
  errors << "_data/writing_categories.yml: duplicate key #{writing_type}" if count > 1
end

writing_categories.map { |category| category["url"] }.tally.each do |url, count|
  errors << "_data/writing_categories.yml: duplicate url #{url}" if count > 1
end

Dir[ROOT.join("_posts/*.{md,markdown}")].sort.each do |filename|
  path = Pathname(filename)
  metadata, body = read_front_matter(path)
  next unless metadata["layout"] == "writing-post"

  %w[title permalink writing_type writing_type_label excerpt].each do |field|
    errors << "#{path.relative_path_from(ROOT)}: missing #{field}" if metadata[field].to_s.strip.empty?
  end

  unless writing_types.include?(metadata["writing_type"])
    errors << "#{path.relative_path_from(ROOT)}: unknown writing_type #{metadata['writing_type'].inspect}"
  end

  if metadata["cover"]
    cover_path = ROOT.join(metadata["cover"].sub(%r{\A/+}, ""))
    errors << "#{path.relative_path_from(ROOT)}: missing cover #{metadata['cover']}" unless cover_path.file?
    errors << "#{path.relative_path_from(ROOT)}: cover_alt is required" if metadata["cover_alt"].to_s.strip.empty?
  end

  if metadata["title_lines"] && !metadata["title_lines"].is_a?(Array)
    errors << "#{path.relative_path_from(ROOT)}: title_lines must be an array"
  end

  source_path = metadata["source_path"]
  next if source_path.to_s.empty?

  source_file = ROOT.join(source_path)
  if !source_file.file? && ROOT.join("Japan_Topophilia").directory?
    errors << "#{path.relative_path_from(ROOT)}: missing local source #{source_path}"
    next
  end
  next unless source_file.file?

  source_lines = preserved_text_lines(source_file.read)
  post_lines = preserved_text_lines(body)
  next if source_lines == post_lines

  mismatch = (0...[source_lines.size, post_lines.size].max).find do |index|
    source_lines[index] != post_lines[index]
  end
  errors << "#{path.relative_path_from(ROOT)}: source text differs at preserved line #{mismatch + 1}"
rescue Psych::SyntaxError, RuntimeError => error
  errors << "#{path.relative_path_from(ROOT)}: #{error.message}"
end

publication_orders = []
Dir[ROOT.join("_publications/*.{md,markdown}")].sort.each do |filename|
  path = Pathname(filename)
  metadata, = read_front_matter(path)

  %w[title publication_status order excerpt].each do |field|
    errors << "#{path.relative_path_from(ROOT)}: missing #{field}" if metadata[field].to_s.strip.empty?
  end

  order = metadata["order"]
  if !order.is_a?(Integer) || order < 1
    errors << "#{path.relative_path_from(ROOT)}: order must be a positive integer"
  else
    publication_orders << [order, path.relative_path_from(ROOT)]
  end
rescue Psych::SyntaxError, RuntimeError => error
  errors << "#{path.relative_path_from(ROOT)}: #{error.message}"
end

publication_orders.group_by(&:first).each do |order, entries|
  next if entries.size == 1

  files = entries.map { |entry| entry.last }.join(", ")
  errors << "_publications: duplicate order #{order} in #{files}"
end

html_files = Dir[SITE_ROOT.join("**/*.html")].sort.map { |filename| Pathname(filename) }
html_files.each do |path|
  document = Nokogiri::HTML(path.read)
  relative_path = path.relative_path_from(SITE_ROOT)

  errors << "#{relative_path}: missing title" if document.at_css("title")&.text.to_s.strip.empty?

  redirect_page = document.at_css('meta[http-equiv="refresh"]')
  errors << "#{relative_path}: missing h1" if !redirect_page && document.at_css("h1").nil?

  ids = document.css("[id]").filter_map { |node| node["id"] unless node["id"].to_s.empty? }
  ids.tally.each do |id, count|
    errors << "#{relative_path}: duplicate id ##{id}" if count > 1
  end

  if document.css("pre code").any? { |code| code.text.include?('class="archive__subtitle"') }
    errors << "#{relative_path}: archive heading was rendered as code"
  end

  document.css("img").each do |image|
    errors << "#{relative_path}: image is missing alt" unless image.key?("alt")
  end

  document.css("[href], [src]").each do |node|
    raw_url = node["href"] || node["src"]
    target = local_target(raw_url, path, SITE_ROOT, site_host)
    next unless target
    next if target_exists?(target)

    errors << "#{relative_path}: broken local reference #{raw_url}"
  end
end

Dir[SITE_ROOT.join("**/*.css")].sort.each do |filename|
  path = Pathname(filename)
  path.read.scan(/url\(\s*['\"]?([^'\")]+)['\"]?\s*\)/).flatten.each do |raw_url|
    target = local_target(raw_url, path, SITE_ROOT, site_host)
    next unless target
    next if target_exists?(target)

    errors << "#{path.relative_path_from(SITE_ROOT)}: broken local reference #{raw_url}"
  end
end

errors = errors.uniq.sort

if errors.empty?
  puts "Site validation passed (#{html_files.size} HTML files, #{writing_types.size} writing categories)."
else
  warn "Site validation failed with #{errors.size} issue(s):"
  errors.each { |error| warn "- #{error}" }
  exit 1
end
