import SwiftUI

struct HeroCard: View {
    var body: some View {
        ZStack(alignment: .bottomLeading) {
            GeometryReader { proxy in
                Image("skiHero")
                    .resizable()
                    .scaledToFill()
                    .frame(
                        width: proxy.size.width,
                        height: proxy.size.height,
                        alignment: .trailing
                    )
                    .clipped()
            }

            LinearGradient(
                colors: [Color.slopeNavy.opacity(0.76), Color.slopeNavy.opacity(0.2), .clear],
                startPoint: .leading,
                endPoint: .trailing
            )

            LinearGradient(
                colors: [.clear, Color.slopeNavy.opacity(0.46)],
                startPoint: .center,
                endPoint: .bottom
            )

            VStack(alignment: .leading, spacing: 7) {
                Text("AI Ski Coach")
                    .font(.system(size: 27, weight: .bold))

                Text("Improve your skiing with AI-powered\nvideo analysis")
                    .font(.system(size: 14, weight: .regular))
                    .lineSpacing(2)
                    .foregroundStyle(.white.opacity(0.92))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.bottom, 18)
        }
        .frame(height: 188)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(.white.opacity(0.18), lineWidth: 1)
        }
        .shadow(color: Color.slopeNavy.opacity(0.12), radius: 12, y: 6)
        .accessibilityElement(children: .combine)
    }
}
