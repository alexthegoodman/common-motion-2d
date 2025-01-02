use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

fn main() -> io::Result<()> {
    // Open the input and output files
    let input_file = File::open("D:/projects/common/common-motion-2d/backup/test.txt")?;
    let mut output_file =
        File::create("D:/projects/common/common-motion-2d/backup/cleaned_test.txt")?;

    // Create a reader for the input file
    let reader = BufReader::new(input_file);

    // Process each line
    for line in reader.lines() {
        let line = line?;

        // Preserve separator lines
        if line.trim() == "---" || line.trim() == "!!!" {
            writeln!(output_file, "{}", line)?;
            continue;
        }

        // Split the line into columns
        let columns: Vec<&str> = line.split(',').map(|col| col.trim()).collect();

        // Check if the line has at least 6 columns
        if columns.len() < 6 {
            continue;
        }

        // Parse the second column and check if it equals 15
        if let Ok(second_value) = columns[1].parse::<f32>() {
            if (second_value - 15.0).abs() < f32::EPSILON {
                continue;
            }
        }

        // Extract the last two columns
        if let (Some(last_but_one), Some(last)) = (
            columns.get(columns.len() - 2),
            columns.get(columns.len() - 1),
        ) {
            writeln!(output_file, "{}, {}", last_but_one, last)?;
        }
    }

    println!("Processing complete. Check output.txt for results.");
    Ok(())
}
