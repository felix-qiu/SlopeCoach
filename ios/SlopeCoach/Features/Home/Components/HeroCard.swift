import SwiftUI

struct HeroCard: View {
    var body: some View {
        ZStack(alignment: .bottomLeading) {
            Image("skiHero")
                .resizable()
                .scaledToFill()
                .frame(height: 218)
                .frame(maxWidth: .infinity)
                .clipped()

            LinearGradient(
                colors: [.clear, Color.slopeNavy.opacity(0.18), Color.slopeNavy.opacity(0.9)],
                startPoint: .top,
                endPoint: .bottom
            )

            VStack(alignment: .leading, spacing: 8) {
                Label("PERSONAL COACHING", systemImage: "sparkles")
                    .font(.system(size: 11, weight: .bold))
                    .tracking(0.8)
                    .foregroundStyle(.white.opacity(0.86))

                Text("AI Ski Coach")
                    .font(.system(size: 29, weight: .bold, design: .rounded))

                Text("Improve your skiing with AI-powered\nvideo analysis")
                    .font(.system(size: 15, weight: .medium))
                    .lineSpacing(3)
                    .foregroundStyle(.white.opacity(0.9))
            }
            .foregroundStyle(.white)
            .padding(20)
        }
        .frame(height: 218)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(.white.opacity(0.18), lineWidth: 1)
        }
        .shadow(color: Color.slopeNavy.opacity(0.16), radius: 16, y: 8)
        .accessibilityElement(children: .combine)
    }
}
