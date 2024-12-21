use rand::Rng;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut file = File::create("D:/projects/common/common-motion-2d/backup/vocab.txt")?;

    // Write individual digits
    // for digit in 0..=9 {
    //     writeln!(file, "{}", digit)?;
    // }

    // Write special characters
    for special in [",", ".", "-"] {
        writeln!(file, "{}", special)?;
    }

    let mut rng = rand::thread_rng();

    for x in 0..3000 {
        // let num = rng.gen_range(10..3000);
        writeln!(file, "{}", x)?;
    }

    for x in -3000..0 {
        // let num = rng.gen_range(10..3000);
        writeln!(file, "{}", x)?;
    }

    for x in 0..3000 {
        // let num = rng.gen_range(10..3000);
        writeln!(file, "{}", x as f32 + 0.5)?;
    }

    for x in -3000..0 {
        // let num = rng.gen_range(10..3000);
        writeln!(file, "{}", x as f32 + 0.5)?;
    }

    // // Generate some random 2-digit numbers
    // for _ in 0..50 {
    //     let num = rng.gen_range(10..100);
    //     writeln!(file, "{}", num)?;
    // }

    // // Generate some random 3-digit numbers
    // for _ in 0..500 {
    //     let num = rng.gen_range(100..2000);
    //     writeln!(file, "{}", num)?;
    // }

    // // Generate some negative numbers
    // for _ in 0..500 {
    //     let num = -rng.gen_range(1..2000);
    //     writeln!(file, "{}", num)?;
    // }

    // // Generate some decimal numbers with .5
    // for _ in 0..10 {
    //     let base = rng.gen_range(0..20) as f32 * 2.5;
    //     writeln!(file, "{:.1}", base)?;
    // }

    // // Generate some number pairs with comma
    // for _ in 0..15 {
    //     let num1 = rng.gen_range(100..1000);
    //     let num2 = rng.gen_range(100..1000);
    //     writeln!(file, "{},{}", num1, num2)?;
    // }

    println!("Generated vocab.txt successfully!");
    Ok(())
}
