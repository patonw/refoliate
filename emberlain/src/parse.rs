use std::sync::Arc;
use tree_sitter::{Node, Query, QueryCapture, QueryCursor, StreamingIterator};

pub mod cb {
    use super::Breadcrumb;
    use std::path::Path;
    use std::sync::Arc;
    use tree_sitter::Query;
    use tree_sitter::Tree;

    pub struct FileMatchArgs<'a> {
        pub file_path: &'a Path,
        pub source: &'a [u8],
        pub tree: &'a Tree,
        pub query: &'a Query,
    }

    #[derive(Debug)]
    pub struct NodeMatchArgs<'a> {
        pub query_match: Arc<Breadcrumb<'a>>,
        pub stack: Vec<Arc<Breadcrumb<'a>>>,
    }
}

use cb::*;

#[derive(Debug)]
pub struct Breadcrumb<'a> {
    pub pattern_index: usize,
    pub captures: Vec<QueryCapture<'a>>,
}

// Admittedly a silly name
pub struct Breadloaf<'a> {
    // file_path: &'a Path,
    pub query: &'a Query,
    pub source_code: &'a [u8],
    // stack: Vec<Arc<Breadcrumb<'a>>>,
}

impl<'a> Breadloaf<'a> {
    pub fn capture_nodes(&self, crumb: &'a Breadcrumb, capture_name: &str) -> Vec<Node> {
        if let Some(idx) = self.query.capture_index_for_name(capture_name) {
            crumb
                .captures
                .iter()
                .filter(|cap| cap.index == idx)
                .map(|cap| cap.node)
                .collect()
        } else {
            vec![]
        }
    }

    pub fn capture_texts(&self, crumb: &'a Breadcrumb, capture_name: &str) -> Vec<&'a str> {
        self.capture_nodes(crumb, capture_name)
            .iter()
            .filter_map(|node| node.utf8_text(self.source_code).ok())
            .collect()
    }

    pub fn capture_text(&self, crumb: &'a Breadcrumb, capture_name: &str) -> Option<&'a str> {
        self.capture_texts(crumb, capture_name).into_iter().next()
    }

    pub fn has_capture(&self, crumb: &Breadcrumb, capture_name: &str) -> bool {
        !self.capture_nodes(crumb, capture_name).is_empty()
    }

    /// Attempts to find the primary match by examining capture names starting with "definition"
    pub fn match_kind(&self, crumb: &Breadcrumb) -> Option<&'a str> {
        crumb
            .captures
            .iter()
            .map(|c| c.index as usize)
            .filter(|i| *i < self.query.capture_names().len())
            .map(|i| self.query.capture_names()[i])
            .filter(|s| s.starts_with("definition."))
            .map(|s| s.trim_start_matches("definition."))
            .next()
    }
}

pub type MatchStack<'a> = Vec<Arc<Breadcrumb<'a>>>;

#[derive(Debug)]
pub enum MatchQueueItem<'a> {
    Node(Node<'a>),
    Breadcrumb(Arc<Breadcrumb<'a>>),
}

#[deprecated(note = "use the async version")]
pub fn sync_process_node(
    node: Node,
    source_code: &[u8],
    query: &Query,
    stack: MatchStack<'_>,
    cb: &impl Fn(NodeMatchArgs),
) {
    let mut qc = QueryCursor::new();
    qc.set_max_start_depth(Some(1)); // mod -> decl list -> funcs

    let mut query_matches = qc.matches(query, node, source_code);

    while let Some(query_match) = query_matches.next() {
        let props = query.property_settings(query_match.pattern_index);
        let recurse = props.iter().any(|p| p.key.as_ref() == "recurse");

        let crumb = Arc::new(Breadcrumb {
            pattern_index: query_match.pattern_index,
            captures: query_match.captures.to_vec(),
        });

        if recurse {
            let mut stack = stack.clone();
            stack.push(crumb);

            let body = query_match
                .captures
                .iter()
                .filter_map(|c| c.node.child_by_field_name(b"body"))
                .next()
                .expect("Cannot find subtree body");

            sync_process_node(body, source_code, query, stack, cb);
        } else {
            cb(NodeMatchArgs {
                query_match: crumb,
                stack: stack.clone(),
            });
        }
    }
}

// Recursion makes sense on modules, but for impls, better to spell them out in the query
pub async fn process_node<'a>(
    node: Node<'a>,
    source_code: &'a [u8],
    query: &'a Query,
    stack: MatchStack<'a>,
    cb: &impl AsyncFn(NodeMatchArgs<'a>),
) {
    // Actual recursion not necessary when managing call stack directly
    // This is currently using pseudo-DFS
    // TODO: Fix the insertion order for real DFS
    let mut queue: Vec<(MatchStack, MatchQueueItem)> = vec![(stack, MatchQueueItem::Node(node))];

    while let Some((stack, item)) = queue.pop() {
        match item {
            MatchQueueItem::Node(node) => {
                // Run queries when we have a raw node
                let mut qc = QueryCursor::new();
                qc.set_max_start_depth(Some(1)); // mod -> decl list -> funcs
                let mut query_matches = qc.matches(query, node, source_code);

                while let Some(query_match) = query_matches.next() {
                    let crumb = Arc::new(Breadcrumb {
                        pattern_index: query_match.pattern_index,
                        captures: query_match.captures.to_vec(),
                    });

                    queue.push((stack.clone(), MatchQueueItem::Breadcrumb(crumb)));
                }
            }
            MatchQueueItem::Breadcrumb(crumb) => {
                // Dispatch logic when acting on breadcrums
                let props = query.property_settings(crumb.pattern_index);
                let recurse = props.iter().any(|p| p.key.as_ref() == "recurse");

                if recurse {
                    // Extend the stack, unwrap the crumb into a node and requeue
                    let mut stack = stack.clone();
                    stack.push(crumb.clone());

                    let body = crumb
                        .captures
                        .iter()
                        .filter_map(|c| c.node.child_by_field_name(b"body"))
                        .next()
                        .expect("Cannot find subtree body");

                    queue.push((stack, MatchQueueItem::Node(body)));
                } else {
                    // Just trigger the callback
                    cb(NodeMatchArgs {
                        query_match: crumb,
                        stack: stack.clone(),
                    })
                    .await;
                }
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use anyhow::anyhow;
    use googletest::prelude::*;
    use tree_sitter::Query;

    use crate::test_utils::*;

    #[tokio::test]
    async fn test_empty_queries() -> anyhow::Result<()> {
        let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

        // Note: seems patterns won't match without a capture
        let query_text = r#""#;
        let query = Query::new(&language, query_text)?;
        dbg!(&query);

        let source_code = r#"
            fn foobar() {
                println!("Hello world");
            }
        "#
        .as_bytes();

        let tree = parser
            .parse(source_code, None)
            .ok_or(anyhow!("Could not parse"))?;

        let root = tree.root_node().child(0).expect("File is empty");
        let counter = Arc::new(AtomicUsize::default());

        process_node(root, source_code, &query, vec![], &async |_it| {
            counter.fetch_add(1, Ordering::Relaxed);
        })
        .await;

        assert_that!(counter.load(Ordering::Relaxed), eq(0));

        Ok(())
    }

    #[tokio::test]
    async fn test_simple_fn() -> anyhow::Result<()> {
        let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

        let query_text = r#"
            (function_item
                name: (identifier) @name.definition.function) @definition.function
        "#;
        let query = Query::new(&language, query_text)?;
        dbg!(&query);

        let source_code = r#"
            fn foobar() {
                println!("Hello world");
            }
        "#
        .as_bytes();

        let tree = parser
            .parse(source_code, None)
            .ok_or(anyhow!("Could not parse"))?;

        let root = tree.root_node().child(0).expect("File is empty");
        let counter = Arc::new(AtomicUsize::default());

        let loaf = Breadloaf {
            source_code,
            query: &query,
        };

        process_node(root, source_code, &query, vec![], &async |it| {
            counter.fetch_add(1, Ordering::Relaxed);

            assert_that!(
                loaf.match_kind(it.query_match.as_ref()),
                some(eq("function"))
            );

            let cap = &it.query_match.captures[0];

            assert_that!(cap.node.utf8_text(source_code).unwrap(), starts_with("fn"));

            assert_that!(
                loaf.capture_text(it.query_match.as_ref(), "name.definition.function"),
                some(eq("foobar"))
            );
        })
        .await;

        assert_that!(counter.load(Ordering::Relaxed), eq(1));

        Ok(())
    }

    #[tokio::test]
    async fn test_recursive_query() -> anyhow::Result<()> {
        let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

        // Note: seems patterns won't match without a capture
        let query_text = r#"
            (function_item
                name: (identifier) @name.definition.function) @definition.function

            (mod_item
                name: (identifier) @name.definition.module
                body: (declaration_list)
                (#set! recurse)) @definition.module
        "#;
        let query = Query::new(&language, query_text)?;
        dbg!(&query);

        let source_code = r#"
            mod foo {
                fn foobar() {}
            }
        "#
        .as_bytes();

        let tree = parser
            .parse(source_code, None)
            .ok_or(anyhow!("Could not parse"))?;

        let root = tree.root_node().child(0).expect("File is empty");
        let counter = Arc::new(AtomicUsize::default());

        let loaf = Breadloaf {
            source_code,
            query: &query,
        };

        process_node(root, source_code, &query, vec![], &async |it| {
            dbg!(&it);

            assert_that!(counter.load(Ordering::Relaxed), eq(0));

            counter.fetch_add(1, Ordering::Relaxed);

            assert_that!(
                loaf.match_kind(it.query_match.as_ref()),
                some(eq("function"))
            );

            let cap = &it.query_match.captures[0];

            assert_that!(cap.node.utf8_text(source_code).unwrap(), starts_with("fn"));

            let parent = it.stack[0].captures[0];
            assert_that!(
                parent.node.utf8_text(source_code).unwrap(),
                starts_with("mod")
            );
        })
        .await;

        Ok(())
    }
}
