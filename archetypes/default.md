---
date: '{{ .Date }}'
draft: false
params:
  math: true
title: '{{ replace .File.ContentBaseName `-` ` ` | title }}'
tags: []
---