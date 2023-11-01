fn it_works() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_it_works() {
        assert!(it_works());
    }
}
