use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::anyhow;
use googletest::prelude::*;
use tree_sitter::Query;

use emberlain::parse::{Breadloaf, process_node};

mod utils;
use utils::*;

#[tokio::test]
async fn it_struct_def() -> anyhow::Result<()> {
    let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

    let query_text = r#"
        (struct_item
            name: (type_identifier) @name.definition.class) @definition.class
    "#;
    let query = Query::new(&language, query_text)?;
    dbg!(&query);

    let source_code = r#"
        struct Foobar {
            foo: String,
            bar: Vec<usize>,
        }
    "#
    .as_bytes();

    let tree = parser
        .parse(source_code, None)
        .ok_or(anyhow!("Could not parse"))?;

    let root = tree.root_node();
    let counter = Arc::new(AtomicUsize::default());

    let loaf = Breadloaf {
        source_code,
        query: &query,
    };

    process_node(root, source_code, &query, vec![], &async |it| {
        dbg!(&it);

        counter.fetch_add(1, Ordering::Relaxed);

        assert_that!(it.query_match.captures.len(), eq(2));
        assert_that!(loaf.match_kind(it.query_match.as_ref()), some(eq("class")));

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.reference.class"),
            none()
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.definition.class"),
            some(eq("Foobar"))
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "definition.class"),
            some(contains_substring("Vec<usize>"))
        );
    })
    .await;

    assert_that!(counter.load(Ordering::Relaxed), eq(1));

    Ok(())
}

#[tokio::test]
async fn it_impl_struct() -> anyhow::Result<()> {
    let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

    // Note: seems patterns won't match without a capture
    let query_text = r#"
        (impl_item
            trait: (type_identifier) @name.reference.interface
            type: (type_identifier) @name.reference.class
            body: (declaration_list
                (function_item
                    name: (identifier) @name.definition.method) @definition.method
            )
        )

        (impl_item
            type: (type_identifier) @name.reference.class
            !trait
            body: (declaration_list
                (function_item
                    name: (identifier) @name.definition.method) @definition.method
            )
        )
    "#;
    let query = Query::new(&language, query_text)?;
    dbg!(&query);

    let source_code = r#"
        impl Foobar {
            pub fn foo() {
                println!("Hello world");
            }

            pub fn bar() {
                println!("Goodbye world");
            }
        }
    "#
    .as_bytes();

    let tree = parser
        .parse(source_code, None)
        .ok_or(anyhow!("Could not parse"))?;

    let root = tree.root_node();
    let counter = Arc::new(AtomicUsize::default());

    let loaf = Breadloaf {
        source_code,
        query: &query,
    };

    process_node(root, source_code, &query, vec![], &async |it| {
        dbg!(&it);

        counter.fetch_add(1, Ordering::Relaxed);

        assert_that!(it.query_match.captures.len(), eq(3));
        assert_that!(loaf.match_kind(it.query_match.as_ref()), some(eq("method")));

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.reference.interface"),
            none()
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.reference.class"),
            some(eq("Foobar"))
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.definition.method"),
            some(any!(eq("foo"), eq("bar")))
        );
    })
    .await;

    assert_that!(counter.load(Ordering::Relaxed), eq(2));

    Ok(())
}

#[tokio::test]
async fn it_impl_trait() -> anyhow::Result<()> {
    let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

    // Note: These patterns will miss associated functions
    let query_text = r#"
        (impl_item
            trait: (type_identifier) @name.reference.interface
            type: (type_identifier) @name.reference.class
            body: (declaration_list
                (function_item
                    name: (identifier) @name.definition.method
                    parameters: (parameters
                        (self_parameter)
                    )
                ) @definition.method
            )
        )

        (impl_item
            type: (type_identifier) @name.reference.class
            !trait
            body: (declaration_list
                (function_item
                    name: (identifier) @name.definition.method
                    parameters: (parameters
                        (self_parameter)
                    )
                ) @definition.method
            )
        )
    "#;
    let query = Query::new(&language, query_text)?;
    dbg!(&query);

    let source_code = r#"
        impl Widget for Foobar {
            pub fn foo(&self, n: usize) {
                println!("Hello world");
            }

            pub fn bar(&self, k: String) {
                println!("Goodbye world");
            }
        }
    "#
    .as_bytes();

    let tree = parser
        .parse(source_code, None)
        .ok_or(anyhow!("Could not parse"))?;

    let root = tree.root_node();
    let counter = Arc::new(AtomicUsize::default());

    let loaf = Breadloaf {
        source_code,
        query: &query,
    };

    process_node(root, source_code, &query, vec![], &async |it| {
        dbg!(&it);

        counter.fetch_add(1, Ordering::Relaxed);

        assert_that!(it.query_match.captures.len(), eq(4));

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.reference.interface"),
            some(eq("Widget"))
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.reference.class"),
            some(eq("Foobar"))
        );

        assert_that!(
            loaf.capture_text(it.query_match.as_ref(), "name.definition.method"),
            some(any!(eq("foo"), eq("bar")))
        );
    })
    .await;

    assert_that!(counter.load(Ordering::Relaxed), eq(2));

    Ok(())
}

// Ensure that free function query doesn't match methods
#[tokio::test]
async fn it_method_not_func() -> anyhow::Result<()> {
    let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

    // Note: seems patterns won't match without a capture
    let query_text = r#"
        (function_item
            name: (identifier) @name.definition.function) @definition.function
    "#;
    let query = Query::new(&language, query_text)?;
    dbg!(&query);

    let source_code = r#"
        impl Foobar {
            pub fn foo() {
                println!("Hello world");
            }
        }
    "#
    .as_bytes();

    let tree = parser
        .parse(source_code, None)
        .ok_or(anyhow!("Could not parse"))?;

    let root = tree.root_node();
    dbg!((root.kind(), root.grammar_name(), root.language().name()));
    let counter = Arc::new(AtomicUsize::default());

    process_node(root, source_code, &query, vec![], &async |it| {
        dbg!(&it);
    })
    .await;

    assert_that!(counter.load(Ordering::Relaxed), eq(0));

    Ok(())
}

#[tokio::test]
async fn it_func_attrs() -> anyhow::Result<()> {
    let (language, mut parser) = aload_language("rust", TREE_SITTER_RUST).await?;

    let query_text = r#"
            ; It seems the docs are wrong:
            ; > https://tree-sitter.github.io/tree-sitter/using-parsers/queries/2-operators.html#grouping-sibling-nodes
            ; The example shows no anchoring '.'
            ; However, omitting this allows any number of intervening elements between siblings.
            (
                (
                    (attribute_item)
                    [(line_comment) (block_comment)]*
                )*
                .
                (function_item
                    name: (identifier) @name.definition.function)
            ) @definition.function
        "#;
    let query = Query::new(&language, query_text)?;
    dbg!(&query);

    let source_code = r#"
            // Ignore this
            #[cfg(feature = "ssr")]
            /// This does stuff
            #[tokio::main]
            // and this maybe
            fn foobar() {
                println!("Hello world");
            }

            // You can't see me
            fn barbaz() { // I has no attribute
                println!("Hello world");
            }
        "#
    .as_bytes();

    let tree = parser
        .parse(source_code, None)
        .ok_or(anyhow!("Could not parse"))?;

    let root = tree.root_node();
    // let root = tree.root_node().child(0).expect("File is empty");
    dbg!((
        root.kind(),
        root.grammar_name(),
        root.language().name(),
        root.to_sexp()
    ));
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

        if loaf.capture_text(it.query_match.as_ref(), "name.definition.function") == Some("foobar")
        {
            assert_that!(
                cap.node.utf8_text(source_code).unwrap(),
                starts_with("#[cfg("),
                "foobar should start with a cfg attribute"
            );
        } else {
            assert_that!(
                cap.node.utf8_text(source_code).unwrap(),
                starts_with("fn"),
                "barbaz should start with fn keyword"
            );
        }
    })
    .await;

    assert_that!(counter.load(Ordering::Relaxed), eq(2));

    Ok(())
}
