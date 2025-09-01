pub fn split_string(input: &str, delimiter: char) -> Vec<String> {
    input.trim()
        .split(delimiter)
        .map(|s| s.to_string())
        .collect()
}