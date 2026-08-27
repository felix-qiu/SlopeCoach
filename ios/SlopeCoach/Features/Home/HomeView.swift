import SwiftUI

struct HomeView: View {
    let router: AppRouter
    @State private var viewModel = HomeViewModel()

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                header
                    .padding(.bottom, 20)

                HeroCard()
                    .padding(.bottom, 16)

                PrimaryButton(title: "Start Analysis", systemImage: "video.badge.plus") {
                    viewModel.startAnalysis(using: router)
                }
                .padding(.bottom, 30)

                VStack(spacing: 14) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Recent Analyses")
                            .font(.system(size: 22, weight: .bold, design: .rounded))
                            .foregroundStyle(Color.slopeNavy)

                        Spacer()

                        Text("Latest activity")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }

                    ForEach(viewModel.recentAnalyses) { analysis in
                        AnalysisCard(analysis: analysis)
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 24)
        }
        .background(Color.slopeBackground.ignoresSafeArea())
        .scrollIndicators(.hidden)
        .toolbar(.hidden, for: .navigationBar)
    }

    private var header: some View {
        HStack(spacing: 11) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.slopePrimary)
                    .frame(width: 42, height: 42)

                Image(systemName: "mountain.2.fill")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white)
            }

            Text("SlopeCoach")
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(Color.slopeNavy)

            Spacer()

            Button(action: {}) {
                ZStack {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 42, height: 42)
                        .shadow(color: .black.opacity(0.07), radius: 8, y: 3)

                    Image(systemName: "person.crop.circle.fill")
                        .font(.system(size: 31))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(Color.slopePrimary, Color.slopePrimary.opacity(0.14))
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open profile")
        }
    }
}
