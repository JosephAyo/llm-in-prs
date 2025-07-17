import ca.usask.cs.text.readability.FleschKincaidReadingEase;

public class ReadabilityCLI {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("Please provide text input as a single argument.");
            System.exit(1);
        }
        // Combine all args in case description has spaces
        String input = String.join(" ", args);

        FleschKincaidReadingEase ease = new FleschKincaidReadingEase(input);
        double score = ease.getReadingEase();
        System.out.println(score);
    }
}
