use std::{borrow::Cow, ops::Deref, sync::Arc};

use anyhow::Result;
use minijinja::{Environment, context};

use crate::{CodeSnippet, LanguageMap};

pub struct Templater<'a> {
    pub env: Environment<'a>,
    pub langmap: Arc<LanguageMap>,
}

impl<'a> Templater<'a> {
    pub fn new(langmap: Arc<LanguageMap>) -> Result<Self> {
        let mut env = Environment::new();

        for (lang, spec) in langmap.as_ref().deref() {
            if let Some(templates) = &spec.templates {
                // TODO: load template on demand
                if let Some(temp) = &templates.class_member {
                    let temp = temp.clone();
                    env.add_template_owned(format!("{lang}::class_member"), temp.clone())?;
                }
                if let Some(temp) = &templates.impl_trait {
                    env.add_template_owned(format!("{lang}::impl_trait"), temp.clone())?;
                }
            }
        }
        Ok(Self { env, langmap })
    }

    pub fn render(&self, snippet: CodeSnippet) -> Result<CodeSnippet> {
        let (lang, _spec) = self.langmap.get_by_path(&snippet.path)?;

        let mut rendered = Cow::Borrowed(&snippet.body);
        match (&snippet.interface, &snippet.class) {
            (Some(trait_type), Some(class_type)) => {
                if let Ok(tmpl) = self.env.get_template(&format!("{lang}::impl_trait"))
                    && let Ok(text) =
                        tmpl.render(context! { trait_type, class_type, code => &snippet.body })
                {
                    rendered = Cow::Owned(text)
                } else if let Ok(out) = render_impl_trait(trait_type, class_type, &snippet.body) {
                    rendered = Cow::Owned(out)
                }
            }
            (None, Some(class_type)) => {
                if let Ok(tmpl) = self.env.get_template(&format!("{lang}::class_member"))
                    && let Ok(text) = tmpl.render(context! { class_type, code => &snippet.body })
                {
                    rendered = Cow::Owned(text)
                } else if let Ok(out) = render_class_member(class_type, &snippet.body) {
                    rendered = Cow::Owned(out)
                }
            }
            _ => {}
        };

        Ok(CodeSnippet {
            // TODO: store as Cow (but requires lifetime on CodeSnippet... moo)
            rendered: rendered.into_owned(),
            ..snippet
        })
    }
}

fn render_impl_trait(trait_type: &str, class_type: &str, code: &str) -> Result<String> {
    let env = Environment::new();
    let tmpl = env.template_from_named_str(
        "impl_trait",
        "Implementation of {{trait_type}} for {{class_type}}:n```\n{{code}}\n```",
    )?;
    Ok(tmpl.render(context! { trait_type, class_type, code })?)
}

fn render_class_member(class_type: &str, code: &str) -> Result<String> {
    let env = Environment::new();
    let tmpl = env.template_from_named_str("class_member", "/// self: {{class_type}}\n{{code}}")?;
    Ok(tmpl.render(context! { class_type, code })?)
}
